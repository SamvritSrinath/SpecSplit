# Core Modules

## `specsplit.core.config`

::: specsplit.core.config
    options:
      show_root_heading: false
      show_source: false
      members:
        - DraftWorkerConfig
        - TargetWorkerConfig
        - OrchestratorConfig
        - load_config_file

## `specsplit.core.telemetry`

::: specsplit.core.telemetry
    options:
      show_root_heading: false
      show_source: false
      members:
        - Stopwatch
        - TelemetrySpan
        - TelemetryEvent
        - TelemetryLogger
        - get_current_context

## `specsplit.core.verification`

::: specsplit.core.verification
    options:
      show_root_heading: false
      show_source: false
      members:
        - VerificationResult
        - verify_greedy_tree
        - verify_stochastic_tree

## `specsplit.core.model_loading`

::: specsplit.core.model_loading
    options:
      show_root_heading: false
      show_source: false
      members:
        - get_checkpoint_dtype
        - get_model_config
        - get_model_vocab_size

## `specsplit.core.cache_utils`

::: specsplit.core.cache_utils
    options:
      show_root_heading: false
      show_source: false
      members:
        - cache_to_legacy
        - cache_supports_crop
        - crop_cache
        - batch_model_caches
        - slice_batch_item_from_cache
        - legacy_to_dynamic_cache

## `specsplit.core.serialization`

::: specsplit.core.serialization
    options:
      show_root_heading: false
      show_source: false
      members:
        - tensor_to_token_ids
        - token_ids_to_tensor
        - logits_to_probs

