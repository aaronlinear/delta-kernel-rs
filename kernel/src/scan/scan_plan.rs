//! Declarative metadata scan plans.
//!
//! [`Scan::build_metadata_scan_plan`] reconciles checkpoint and commit actions into live adds,
//! applying metadata pruning before newest-action-wins replay.

use std::borrow::Cow;
use std::sync::{Arc, LazyLock};

use url::Url;

use super::data_skipping::as_sql_data_skipping_predicate_with_stats_columns;
use super::state_info::StateInfo;
use super::{PhysicalPredicate, Scan};
use crate::actions::deletion_vector::DeletionVectorDescriptor;
use crate::actions::{
    ADD_NAME, ADD_SCHEMA, REMOVE_FIELD, SIDECAR_FIELD, SIDECAR_NAME, STATS_PARSED,
};
use crate::checkpoint::{CheckpointShape, CheckpointType};
use crate::expressions::{
    col, column_name, joined_column_expr, lit, null_lit, ColumnName, Expression as Expr,
    ExpressionRef, Predicate,
};
use crate::plans::ir::nodes::{FileType, Load, LoadColumnFileMeta, ScanFile};
use crate::plans::ir::plan::Plan;
use crate::scan::log_replay::{PARTITION_VALUES_PARSED_NAME, STATS_PARSED_NAME};
use crate::schema::{
    lazy_schema_ref, schema, schema_ref, DataType, SchemaRef, SchemaStructPatchBuilder,
    StructField, StructType, ToSchema as _,
};
use crate::struct_patch::ProjectionStructPatchBuilder;
use crate::transforms::{transform_output_type, ExpressionTransform};
use crate::utils::{CollectInto, FoldWithOption as _};
use crate::{DeltaResult, Error, PlanBuilder};

// === Internal column names ===

// Both add and remove provide path + DV (storageType, pathOrInlineDv, offset) columns. We
// materialize them as one top-level `file_action_key` column that are used by the plan's
// aggregate and anti-join operators.
const FILE_ACTION_KEY: &str = "file_action_key";
const STATS: &str = "stats";
const PARTITION_VALUES: &str = "partitionValues";
const PARTITION_VALUES_PARSED: &str = "partitionValues_parsed";
// Generated partition pruning predicates reference this to retain removes.
const IS_ADD: &str = "is_add";
const VERSION: &str = "version";

/// Execution schema for Add actions extracted from the nullable top-level action union.
///
/// A required child of a nullable struct is nullable after extraction: rows belonging to another
/// action have a null `add` parent and therefore null extracted children. Keep nested child types
/// unchanged because their own parent null bitmap still carries that nullability.
static EXTRACTED_ADD_SCHEMA: LazyLock<StructType> = LazyLock::new(|| {
    StructType::new_unchecked(ADD_SCHEMA.fields().cloned().map(|mut field| {
        field.nullable = true;
        field
    }))
});
static EXTRACTED_ADD_FIELD: LazyLock<StructField> =
    LazyLock::new(|| StructField::nullable(ADD_NAME, EXTRACTED_ADD_SCHEMA.clone()));

impl Scan {
    /// Build the live-add metadata plan from checkpoint and commit actions.
    ///
    /// Returns `None` for an empty result or a statically false predicate.
    pub(super) fn build_metadata_scan_plan(
        &self,
        shape: &CheckpointShape,
    ) -> DeltaResult<Option<Plan>> {
        let state = &self.state_info;
        // A statically-unsatisfiable predicate (e.g. `x > 10 AND FALSE`) skips the whole table.
        if state.physical_predicate == PhysicalPredicate::StaticSkipAll {
            return Ok(None);
        }

        let prune = stats_skipping_predicate(state);
        let prune = prune.as_ref();

        // The output `add` after reparsing `stats`/`partitionValues`: shared by the commit arm's
        // dedup carrier and both terminal `{ add }` projections, so every arm agrees on the
        // union schema.
        let add_field = self.normalized_add_field()?;
        let (output_expr, output_schema) = self.metadata_output_projection(&add_field)?;

        let commit_actions = self.commit_arm()?.try_fold_with(prune, |p, prune| {
            // We filter so that:
            // * All remove actions are kept
            // * Add actions that do not match the partition pruning or stats predicate are removed.
            //
            // NOTE: It is important that add actions are filtered by the partition predicate
            // because partition filtering may not be applied on data rows. On the other
            // hand, failing to skip based on data columns is safe because the data
            // predicate will also be evaluated on data rows. Thus it is crucial that we partition
            // prune adds here.
            //
            // NOTE: It is not safe to prune remove actions using the partition filter. This is
            // because a NULL result for `remove.partitionValues.partCol` may be due to
            // `remove.partitionValues` being NULL, or it may be from `partCol` being
            // NULL. Thus, we simply do not prune removes.
            p.filter(Predicate::or(col!("add").is_null(), prune.clone()))
        })?;

        let deduped_commit =
            commit_actions.aggregate_by([ColumnName::new([FILE_ACTION_KEY])], |a| {
                // Each group with a non-null key contains the adds and removes for a given file.
                // Winning adds pass through unchanged while winning removes produce NULL.
                // Non-file actions have a NULL key and map to their own NULL group.
                a.max_non_null_by(
                    ColumnName::new([ADD_NAME]),
                    ColumnName::new([FILE_ACTION_KEY]),
                    ColumnName::new([VERSION]),
                )
            })?;

        let checkpoint_adds = self
            .checkpoint_arm(shape)?
            .try_fold_with(prune, |p, prune| p.filter(prune.clone()))?;

        let checkpoint_live_adds = checkpoint_adds
            .anti_join(
                deduped_commit.clone(),
                [ColumnName::new([FILE_ACTION_KEY])],
                [ColumnName::new([FILE_ACTION_KEY])],
            )?
            .project(output_expr.clone(), output_schema.clone())?;

        let commit_live_adds = deduped_commit
            .filter(col!("add").is_not_null())?
            .project(output_expr, output_schema)?;

        PlanBuilder::union_all([commit_live_adds, checkpoint_live_adds])?.build_opt()
    }

    /// Build normalized checkpoint adds. Returns an empty relation when no checkpoint exists.
    ///
    /// ## SQL equivalent:
    //
    /// SELECT STRUCT(
    ///          add.* EXCEPT (
    ///            stats_parsed, partitionValues_parsed
    ///          ),
    ///          add.stats_parsed AS stats_parsed,
    ///          MAP_TO_STRUCT(add.partitionValues, physical_partitions) AS partitionValues_parsed
    ///        ) AS add,
    ///        version, add.path IS NOT NULL AS is_add, file_key(add) AS key
    /// FROM checkpoint_actions
    /// WHERE add.path IS NOT NULL
    ///
    /// When the checkpoint lacks native parsed stats, `FROM_JSON(add.stats, physical_stats)`
    /// replaces `add.stats_parsed` above. A parsed field is omitted when its schema is absent.
    fn checkpoint_arm(&self, shape: &CheckpointShape) -> DeltaResult<PlanBuilder> {
        let log_segment = self.snapshot.log_segment();
        let physical_stats = self.state_info.physical_stats_schema.as_ref();
        let physical_partitions = self.state_info.physical_partition_schema.as_ref();
        let source_physical_stats = shape.parsed_stats_schema.as_ref();
        let checkpoint = log_segment.checkpoint_version_tagged_scan_files()?;

        let actions = match (&shape.checkpoint_type, checkpoint) {
            (CheckpointType::Leaf, Some((FileType::Parquet, parts))) => {
                let schema = parquet_read_schema(source_physical_stats, None)?;
                PlanBuilder::scan_parquet(parts, &[VERSION], schema)
            }
            (CheckpointType::Leaf, Some((FileType::Json, parts))) => {
                PlanBuilder::scan_json(
                    parts,
                    &[VERSION],
                    json_read_schema(/* include_remove */ false),
                )
            }
            (CheckpointType::Manifest, Some((file_type, parts))) => {
                let schema = parquet_read_schema(source_physical_stats, None)?;
                match log_segment.checkpoint_hint_version_tagged_sidecar_scan_files()? {
                    Some(sidecars) => PlanBuilder::scan_parquet(sidecars, &[VERSION], schema),
                    // Without a complete hint, load the sidecars referenced by the manifest.
                    None => sidecar_actions(file_type, parts, schema, &log_segment.log_root),
                }
            }
            (CheckpointType::None, _) | (_, None) => {
                PlanBuilder::values(json_read_schema(/* include_remove */ false), vec![])
            }
        }?;

        actions
            .filter(col!("add.path").is_not_null())?
            .project_patch(|patch| {
                patch
                    .with_parsed_add_stats(physical_stats)
                    .with_parsed_add_partition_values(physical_partitions)
                    .append(
                        StructField::not_null(IS_ADD, DataType::BOOLEAN),
                        Expr::from(col!("add.path").is_not_null()),
                    )
                    .append(
                        FILE_ACTION_KEY_FIELD.clone(),
                        file_action_key_expr(|col| joined_column_expr!("add", col)),
                    )
            })
    }

    /// Build the normalized commit JSON arm.
    ///
    /// ## SQL equivalent:
    ///
    /// SELECT STRUCT(
    ///          add.* EXCEPT (stats_parsed, partitionValues_parsed),
    ///          FROM_JSON(add.stats, physical_stats) AS stats_parsed,
    ///          MAP_TO_STRUCT(add.partitionValues, physical_partitions) AS partitionValues_parsed
    ///        ) AS add,
    ///        remove, version, add.path IS NOT NULL AS is_add,
    ///        file_key(COALESCE(add, remove)) AS key
    /// FROM json_commits
    /// WHERE add.path IS NOT NULL OR remove.path IS NOT NULL
    ///
    /// A parsed field is omitted when its schema is absent.
    fn commit_arm(&self) -> DeltaResult<PlanBuilder> {
        let log_segment = self.snapshot.log_segment();
        let commit_files = log_segment.commit_cover_version_tagged_scan_files()?;
        PlanBuilder::scan_json(commit_files, &[VERSION], json_read_schema(true))?
            .filter(Predicate::or(
                col!("add.path").is_not_null(),
                col!("remove.path").is_not_null(),
            ))?
            .project_patch(|patch| {
                // Commits never carry source-native parsed columns, so normalize from the raw
                // encodings.
                patch
                    .with_parsed_add_stats(self.state_info.physical_stats_schema.as_ref())
                    .with_parsed_add_partition_values(
                        self.state_info.physical_partition_schema.as_ref(),
                    )
                    .append(
                        StructField::not_null(IS_ADD, DataType::BOOLEAN),
                        Expr::from(col!("add.path").is_not_null()),
                    )
                    .append(
                        FILE_ACTION_KEY_FIELD.clone(),
                        file_action_key_expr(|col| {
                            Expr::coalesce([
                                joined_column_expr!("add", col),
                                joined_column_expr!("remove", col),
                            ])
                        }),
                    )
            })
    }

    fn normalized_add_field(&self) -> DeltaResult<StructField> {
        let physical_stats_schema = self.state_info.physical_stats_schema.as_ref();
        let physical_partition_schema = self.state_info.physical_partition_schema.as_ref();
        let patch = SchemaStructPatchBuilder::new()
            .fold_with(physical_stats_schema, |patch, schema| {
                patch.append(StructField::nullable(STATS_PARSED, schema.as_ref().clone()))
            })
            .fold_with(physical_partition_schema, |patch, schema| {
                patch.append(StructField::nullable(
                    PARTITION_VALUES_PARSED,
                    schema.as_ref().clone(),
                ))
            });
        Ok(StructField::nullable(
            ADD_NAME,
            patch.build(&EXTRACTED_ADD_SCHEMA)?,
        ))
    }

    /// Builds the output projection for requested stats and partition values. The base of this
    /// transformation is constructed by [`Self::normalized_add_field`].
    ///
    /// The output schema is:
    /// ```text
    /// add: struct<
    ///   path: string,
    ///   partitionValues: map<string, string>,
    ///   size: long,
    ///   modificationTime: long,
    ///   dataChange: boolean,
    ///   stats: string,                         // when JSON stats are requested
    ///   tags: map<string, string>,
    ///   deletionVector: struct<...>,
    ///   baseRowId: long,
    ///   defaultRowCommitVersion: long,
    ///   clusteringProvider: string,
    ///   stats_parsed: struct<...>,             // when parsed stats are requested
    ///   partitionValues_parsed: struct<...>,   // when parsed partition values are requested
    /// >
    /// ```
    /// Stats output may contain neither representation, JSON only, parsed only, or both. Parsed
    /// partition values are selected independently and omitted for unpartitioned tables. Fields
    /// needed only for pruning are omitted.
    fn metadata_output_projection(
        &self,
        add_field: &StructField,
    ) -> DeltaResult<(ExpressionRef, SchemaRef)> {
        let input_schema = schema_ref! { (add_field.clone()) };
        let has_stats_parsed = input_schema.contains_col([ADD_NAME, STATS_PARSED_NAME]);
        let projection = ProjectionStructPatchBuilder::new_nested(&input_schema, [ADD_NAME]);

        // JSON stats output. `StatsOptions` allows JSON only, parsed only, both, or neither.
        let has_json_stats = input_schema.contains_col([ADD_NAME, STATS]);
        let projection = match (self.stats.synthesize_json, has_json_stats) {
            (true, true) | (false, false) => projection,
            (true, false) => {
                return Err(Error::internal_error(
                    "JSON stats were requested, but add.stats is missing from the metadata schema",
                ));
            }
            (false, true) => projection.drop(STATS),
        };

        // Parsed stats output.
        let projection = match (self.physical_stats_output_schema.as_ref(), has_stats_parsed) {
            (Some(physical_stats), _) => projection.replace(
                STATS_PARSED,
                StructField::nullable(STATS_PARSED, physical_stats.as_ref().clone()),
                project_nested_struct_to_schema([ADD_NAME, STATS_PARSED_NAME], physical_stats),
            ),
            (None, true) => projection.drop(STATS_PARSED),
            (None, false) => projection,
        };

        // Parsed partition-values output.
        let has_partition_values_parsed =
            input_schema.contains_col([ADD_NAME, PARTITION_VALUES_PARSED_NAME]);
        let physical_partitions = self
            .partition_values
            .parsed_struct
            .then_some(self.state_info.physical_partition_schema.as_ref())
            .flatten();
        let projection = match (physical_partitions, has_partition_values_parsed) {
            (Some(schema), true) => projection.replace(
                PARTITION_VALUES_PARSED,
                StructField::nullable(PARTITION_VALUES_PARSED, schema.as_ref().clone()),
                project_nested_struct_to_schema([ADD_NAME, PARTITION_VALUES_PARSED_NAME], schema),
            ),
            (Some(_), false) => {
                return Err(Error::internal_error(
                    "parsed partition values were requested, but add.partitionValues_parsed is \
                     missing",
                ));
            }
            (None, true) => projection.drop(PARTITION_VALUES_PARSED),
            (None, false) => projection,
        };

        let (add_schema, add_expr) = projection.build()?;
        let schema = schema_ref! { nullable ADD_NAME: (add_schema.as_ref().clone()) };
        Ok((Arc::new(Expr::struct_from([add_expr])), schema))
    }
}

/// Read actions from V2 checkpoint sidecars.
fn sidecar_actions(
    file_type: FileType,
    root_parts: Vec<ScanFile>,
    action_schema: SchemaRef,
    log_root: &Url,
) -> DeltaResult<PlanBuilder> {
    const FILE_PATH: &str = "path";
    const FILE_SIZE: &str = "size";
    const NUM_RECORDS: &str = "num_records";
    const DV: &str = "dv";
    const SIDECAR_SIZE: &str = "sizeInBytes";

    static SIDECAR_FILE_META_SCHEMA: LazyLock<SchemaRef> = lazy_schema_ref! {
        not_null FILE_PATH: STRING,
        not_null FILE_SIZE: LONG,
        nullable NUM_RECORDS: LONG,
        nullable DV: (DeletionVectorDescriptor::to_schema()),
        nullable VERSION: LONG,
    };

    static SIDECAR_READ_SCHEMA: LazyLock<SchemaRef> = lazy_schema_ref! {
        (&SIDECAR_FIELD),
        nullable VERSION: LONG,
    };

    let scan = match file_type {
        FileType::Json => PlanBuilder::scan_json,
        FileType::Parquet => PlanBuilder::scan_parquet,
    };
    let sidecar_files = scan(root_parts, &[VERSION], SIDECAR_READ_SCHEMA.clone())?
        .filter(col!(SIDECAR_NAME, FILE_PATH).is_not_null())?
        .project(
            Expr::struct_from([
                col!(SIDECAR_NAME, FILE_PATH),
                col!(SIDECAR_NAME, SIDECAR_SIZE),
                null_lit(DataType::LONG),
                null_lit(DeletionVectorDescriptor::to_schema()),
                col!(VERSION),
            ]),
            SIDECAR_FILE_META_SCHEMA.clone(),
        )?;

    let load = Load::new(
        action_schema,
        FileType::Parquet,
        LoadColumnFileMeta::new(
            ColumnName::new([FILE_PATH]),
            ColumnName::new([FILE_SIZE]),
            ColumnName::new([NUM_RECORDS]),
        ),
        ColumnName::new([DV]),
    )
    .with_base_url(log_root.join("_sidecars/")?)
    .with_file_constant_columns([VERSION]);

    sidecar_files.load(load)
}

// === Helpers ===

/// Read schema for JSON actions tagged with their log version.
/// Commits include removes; JSON checkpoint leaves do not.
fn json_read_schema(include_remove: bool) -> SchemaRef {
    schema_ref! {
        (&EXTRACTED_ADD_FIELD),
        ..(include_remove.then_some(&REMOVE_FIELD)),
        nullable VERSION: LONG,
    }
}

/// Read schema for parquet add actions.
fn parquet_read_schema(
    physical_stats: Option<&SchemaRef>,
    physical_partitions: Option<&SchemaRef>,
) -> DeltaResult<SchemaRef> {
    let add_patch = SchemaStructPatchBuilder::new()
        .fold_with(physical_stats, |patch, schema| {
            patch.append(StructField::nullable(STATS_PARSED, schema.as_ref().clone()))
        })
        .fold_with(physical_partitions, |patch, schema| {
            patch.append(StructField::nullable(
                PARTITION_VALUES_PARSED,
                schema.as_ref().clone(),
            ))
        });
    Ok(schema_ref! {
        nullable ADD_NAME: (add_patch.build(&EXTRACTED_ADD_SCHEMA)?),
        nullable VERSION: LONG,
    })
}

/// File identity used for replay.
static FILE_ACTION_KEY_FIELD: LazyLock<StructField> = LazyLock::new(|| {
    let schema = schema! {
        nullable "path": STRING,
        nullable "deletionVector": {
            nullable "storageType": STRING,
            nullable "pathOrInlineDv": STRING,
            nullable "offset": INTEGER,
        },
    };
    StructField::nullable(FILE_ACTION_KEY, schema)
});

/// Build a file identity from path and deletion vector.
fn file_action_key_expr(key_col_expr: impl Fn(ColumnName) -> Expr) -> Expr {
    let storage_type = key_col_expr(column_name!("deletionVector.storageType"));
    Expr::struct_from([
        key_col_expr(column_name!("path")),
        Expr::struct_with_nullability_from(
            [
                storage_type.clone(),
                key_col_expr(column_name!("deletionVector.pathOrInlineDv")),
                key_col_expr(column_name!("deletionVector.offset")),
            ],
            Expr::from_pred(storage_type.is_not_null()),
        ),
    ])
}

trait ProjectionStructPatchBuilderExt<'a> {
    /// Parses add stats, preferring a compatible parsed field.
    ///
    /// When `physical_stats` is present, the input must contain either
    /// `add.stats_parsed` or the fallback `add.stats` JSON field.
    fn with_parsed_add_stats(self, physical_stats: Option<&SchemaRef>) -> Self;

    /// Parses add partition values, preferring a compatible parsed field.
    fn with_parsed_add_partition_values(self, physical_partitions: Option<&SchemaRef>) -> Self;
}

impl<'a> ProjectionStructPatchBuilderExt<'a> for ProjectionStructPatchBuilder<'a> {
    fn with_parsed_add_stats(self, physical_stats: Option<&SchemaRef>) -> Self {
        let has_stats_parsed = self
            .input_schema()
            .contains_col([ADD_NAME, STATS_PARSED_NAME]);
        let add = [ADD_NAME];
        match physical_stats {
            Some(schema) => {
                let field = StructField::nullable(STATS_PARSED, schema.as_ref().clone());
                let expr = Expr::parse_json(col!("add.stats"), Arc::clone(schema));
                if has_stats_parsed {
                    self
                } else {
                    self.append_at(add, field, expr)
                }
            }
            None => self,
        }
    }

    fn with_parsed_add_partition_values(self, physical_partitions: Option<&SchemaRef>) -> Self {
        let has_partition_values_parsed = self
            .input_schema()
            .contains_col([ADD_NAME, PARTITION_VALUES_PARSED_NAME]);
        let add = [ADD_NAME];
        match physical_partitions {
            Some(schema) => {
                let field = StructField::nullable(PARTITION_VALUES_PARSED, schema.as_ref().clone());
                let expr = Expr::map_to_struct(col!(ADD_NAME, PARTITION_VALUES));
                if has_partition_values_parsed {
                    let expr = Expr::coalesce([col!(ADD_NAME, PARTITION_VALUES_PARSED), expr]);
                    self.replace_at(add, PARTITION_VALUES_PARSED, field, expr)
                } else {
                    self.append_at(add, field, expr)
                }
            }
            None => self,
        }
    }
}

/// Rebuilds `root` to match a narrowed schema while preserving a null parent struct. A direct
/// column reference would retain fields not requested by the caller.
fn project_nested_struct_to_schema(
    root: impl CollectInto<ColumnName>,
    schema: &StructType,
) -> Expr {
    let root = root.collect_into();
    let fields = schema.fields().map(|field| {
        let column = root.join(&ColumnName::new([field.name()]));
        match field.data_type() {
            DataType::Struct(schema) => project_nested_struct_to_schema(column, schema),
            _ => Expr::from(column),
        }
    });
    Expr::struct_with_nullability_from(
        fields,
        Expr::from_pred(Expr::from(root.clone()).is_not_null()),
    )
}

/// Build the metadata pruning predicate, or `None` when no pruning is possible.
fn stats_skipping_predicate(state: &StateInfo) -> Option<Predicate> {
    /// Re-roots metadata columns under `add`.
    struct MetadataSkippingColumnPrefixer;

    impl<'a> ExpressionTransform<'a> for MetadataSkippingColumnPrefixer {
        transform_output_type!(|'a, T| Cow<'a, T>);

        fn transform_expr_column(&mut self, name: &'a ColumnName) -> Cow<'a, ColumnName> {
            let path = name.path();
            let replacement_root = match path.first().map(String::as_str) {
                Some(STATS_PARSED) => [ADD_NAME, STATS_PARSED],
                Some(PARTITION_VALUES_PARSED) => [ADD_NAME, PARTITION_VALUES_PARSED],
                _ => return Cow::Borrowed(name),
            };
            Cow::Owned(ColumnName::new(
                replacement_root
                    .into_iter()
                    .map(str::to_string)
                    .chain(path.iter().skip(1).cloned()),
            ))
        }
    }

    let PhysicalPredicate::Some(pred, _) = &state.physical_predicate else {
        return None;
    };
    let partition_column_names = state
        .physical_partition_schema
        .iter()
        .flat_map(|s| s.fields().map(|f| f.name().to_string()))
        .collect();
    let skipping = as_sql_data_skipping_predicate_with_stats_columns(
        pred,
        &partition_column_names,
        &state.physical_stats_columns,
    )?;
    // A null skipping verdict means the available metadata cannot prove the file is skippable.
    let skipping = Predicate::distinct(skipping, lit(false));
    let mut prefixer = MetadataSkippingColumnPrefixer;
    Some(prefixer.transform_pred(&skipping).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_add_children_are_nullable() {
        assert!(
            EXTRACTED_ADD_SCHEMA.fields().all(StructField::is_nullable),
            "children extracted from the nullable add action must be nullable"
        );
    }

    #[test]
    fn file_action_key_children_are_nullable() {
        let DataType::Struct(key) = FILE_ACTION_KEY_FIELD.data_type() else {
            panic!("file action key must be a struct")
        };
        assert!(key.field("path").unwrap().is_nullable());

        let DataType::Struct(deletion_vector) = key.field("deletionVector").unwrap().data_type()
        else {
            panic!("file action key deletion vector must be a struct")
        };
        assert!(deletion_vector.fields().all(StructField::is_nullable));
    }
}
