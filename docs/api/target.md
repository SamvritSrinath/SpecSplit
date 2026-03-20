# Target Worker API

## `specsplit.workers.target.engine`

::: specsplit.workers.target.engine
    options:
      show_root_heading: false
      show_source: false
      members:
        - CacheDesyncError
        - VerificationResult
        - KVCacheState
        - TargetEngine

## `specsplit.workers.target.kv_cache`

::: specsplit.workers.target.kv_cache
    options:
      show_root_heading: false
      show_source: false
      members:
        - StaticKVCache
        - VirtualKVCache

## `specsplit.workers.target.tree_attn`

::: specsplit.workers.target.tree_attn
    options:
      show_root_heading: false
      show_source: false
      members:
        - build_tree_attention
        - bool_mask_to_float

## `specsplit.workers.target.service`

::: specsplit.workers.target.service
    options:
      show_root_heading: false
      show_source: false
      members:
        - TargetServiceServicer
        - serve

