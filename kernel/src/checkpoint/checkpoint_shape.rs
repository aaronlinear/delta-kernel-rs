//! Resolves a checkpoint shape. This falls into the following cases: no checkpoint,
//! leaf (file actions inline, including multi-part), or manifest (which references sidecar files).
//! When stats are requested, also reports whether the checkpoint has compatible parsed stats.
//! Driven through a [`PlanExecutor`].

// No in-crate caller yet; following PRs will use this.
#![allow(dead_code)]

use url::Url;

use crate::actions::visitors::SidecarVisitor;
use crate::actions::SIDECAR_NAME;
use crate::engine_data::RowVisitor;
use crate::log_segment::LogSegment;
use crate::plans::ir::nodes::FileType;
use crate::plans::{Operation, PlanBuilder, PlanExecutor};
use crate::schema::SchemaRef;
use crate::snapshot::Snapshot;
use crate::{DeltaResult, FileMeta};

/// Topology of a checkpoint: where the `add` / `remove` actions live.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CheckpointType {
    /// No checkpoint files.
    None,
    /// File actions inline. Classic V1, inline V2, and multi-part V1.
    Leaf,
    /// V2 manifest: checkpoint references sidecar files holding the file actions.
    Manifest,
}

/// A snapshot's resolved checkpoint type and parsed-stats schema.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CheckpointShape {
    /// What kind of checkpoint this is.
    pub(crate) checkpoint_type: CheckpointType,
    /// The requested stats schema when the checkpoint has a compatible `add.stats_parsed` struct
    /// to read it from; `None` when stats were not requested or no compatible parsed stats exist.
    pub(crate) parsed_stats_schema: Option<SchemaRef>,
}

impl CheckpointShape {
    /// Resolve `snapshot`'s checkpoint shape. Determines the checkpoint type and, when
    /// `stats_schema` is `Some`, whether the checkpoint contains parsed stats compatible with it.
    pub(crate) fn try_new(
        exec: &dyn PlanExecutor,
        snapshot: &Snapshot,
        stats_schema: Option<&SchemaRef>,
    ) -> DeltaResult<CheckpointShape> {
        let segment = snapshot.log_segment();

        let (root_checkpoint, file_type) = match segment.listed.checkpoint_parts.first() {
            Some(checkpoint) if checkpoint.extension == "json" => {
                (&checkpoint.location, FileType::Json)
            }
            Some(checkpoint) => (&checkpoint.location, FileType::Parquet),
            None => {
                return Ok(CheckpointShape {
                    checkpoint_type: CheckpointType::None,
                    parsed_stats_schema: None,
                })
            }
        };

        // Classify from a V2 checkpoint's `_last_checkpoint` hint when possible, else inspect the
        // file.
        if let Some(shape) =
            Self::from_v2_checkpoint_hint(exec, segment, root_checkpoint, file_type, stats_schema)?
        {
            return Ok(shape);
        }

        // A checkpoint with sidecars is a manifest, one without is a leaf.
        match file_type {
            FileType::Parquet => {
                let cp_schema = match segment.checkpoint_schema() {
                    Some(schema) => schema,
                    None => exec.read_parquet_footer(root_checkpoint.clone())?.schema,
                };
                // No `sidecar` column means the file actions are inline, so this is a leaf.
                if !cp_schema.contains(SIDECAR_NAME) {
                    return Ok(Self::try_new_leaf(Some(cp_schema), stats_schema));
                }
                // The `sidecar` column may still be all-null (not a manifest), so scan it to
                // confirm whether a sidecar is actually present.
                match collect_single_sidecar(exec, root_checkpoint, file_type, &segment.log_root)? {
                    Some(sidecar) => Self::try_new_manifest(exec, sidecar, stats_schema),
                    None => Ok(Self::try_new_leaf(Some(cp_schema), stats_schema)),
                }
            }
            // A JSON checkpoint has no footer schema to inspect, so try to collect a sidecar to
            // decide if it is a manifest or a leaf. A JSON leaf has no readable schema,
            // hence no parsed stats.
            FileType::Json => {
                match collect_single_sidecar(exec, root_checkpoint, file_type, &segment.log_root)? {
                    Some(sidecar) => Self::try_new_manifest(exec, sidecar, stats_schema),
                    None => Ok(Self::try_new_leaf(None, stats_schema)),
                }
            }
        }
    }

    /// Classify the checkpoint from its `_last_checkpoint` sidecar hint, without reading the
    /// checkpoint file. Returns `None` when the hint is absent (or was trimmed away), leaving the
    /// caller to inspect the file. A non-empty sidecar list is a manifest; an empty list is a leaf
    /// (the writer emits an empty list only for a leaf, and trims an oversized manifest to absent,
    /// never to empty).
    fn from_v2_checkpoint_hint(
        exec: &dyn PlanExecutor,
        segment: &LogSegment,
        root_checkpoint: &FileMeta,
        file_type: FileType,
        stats_schema: Option<&SchemaRef>,
    ) -> DeltaResult<Option<CheckpointShape>> {
        match segment.checkpoint_sidecars() {
            Some([sidecar, ..]) => {
                let sidecar_meta = sidecar.to_filemeta(&segment.log_root)?;
                let result = Self::try_new_manifest(exec, sidecar_meta, stats_schema)?;
                Ok(Some(result))
            }
            Some([]) => {
                // A parquet leaf's stats live in its own schema; read it only when stats are
                // requested. A JSON leaf has no readable schema.
                let leaf_schema = match file_type {
                    FileType::Parquet if stats_schema.is_some() => {
                        Some(match segment.checkpoint_schema() {
                            Some(schema) => schema,
                            None => exec.read_parquet_footer(root_checkpoint.clone())?.schema,
                        })
                    }
                    _ => None,
                };
                Ok(Some(Self::try_new_leaf(leaf_schema, stats_schema)))
            }
            None => Ok(None),
        }
    }

    /// Build the shape for a manifest checkpoint. Its file actions and their stats live in the
    /// sidecars, so probe a sidecar's (parquet) footer -- but only when stats were requested. All
    /// sidecars of a checkpoint share one schema, so probing the first is sufficient.
    fn try_new_manifest(
        exec: &dyn PlanExecutor,
        sidecar: FileMeta,
        stats_schema: Option<&SchemaRef>,
    ) -> DeltaResult<CheckpointShape> {
        let parsed_stats_schema = match stats_schema {
            Some(stats_schema) => {
                let footer_schema = exec.read_parquet_footer(sidecar)?.schema;
                LogSegment::schema_has_compatible_stats_parsed(footer_schema.as_ref(), stats_schema)
                    .then(|| stats_schema.clone())
            }
            None => None,
        };
        Ok(CheckpointShape {
            checkpoint_type: CheckpointType::Manifest,
            parsed_stats_schema,
        })
    }

    /// Build the shape for a leaf checkpoint. Its file actions are inline, so `leaf_schema` carries
    /// their stats (`None` for a JSON leaf, which has no readable footer schema).
    fn try_new_leaf(
        leaf_schema: Option<SchemaRef>,
        stats_schema: Option<&SchemaRef>,
    ) -> CheckpointShape {
        let parsed_stats_schema = stats_schema.filter(|stats_schema| {
            leaf_schema.as_ref().is_some_and(|leaf_schema| {
                LogSegment::schema_has_compatible_stats_parsed(leaf_schema.as_ref(), stats_schema)
            })
        });
        CheckpointShape {
            checkpoint_type: CheckpointType::Leaf,
            parsed_stats_schema: parsed_stats_schema.cloned(),
        }
    }
}

/// Read the checkpoint `file`'s `sidecar` column, returning the first referenced sidecar's
/// [`FileMeta`] (enough to classify and probe; not a full enumeration).
fn collect_single_sidecar(
    exec: &dyn PlanExecutor,
    file: &FileMeta,
    file_format: FileType,
    log_root: &Url,
) -> DeltaResult<Option<FileMeta>> {
    let read_schema = LogSegment::sidecar_read_schema();
    // No file-constant columns: the sidecar column is read directly from each file.
    let plan = match file_format {
        FileType::Parquet => PlanBuilder::scan_parquet([file.clone()], &[], read_schema),
        FileType::Json => PlanBuilder::scan_json([file.clone()], &[], read_schema),
    }?
    .build()?;
    let data = exec.execute_op(Operation::QueryPlan(plan))?.into_data()?;

    let mut visitor = SidecarVisitor::default();
    for batch in data {
        visitor.visit_rows_of(batch?.as_ref())?;
        if !visitor.sidecars.is_empty() {
            break;
        }
    }
    match visitor.sidecars.first() {
        Some(sidecar) => Ok(Some(sidecar.to_filemeta(log_root)?)),
        None => Ok(None),
    }
}
