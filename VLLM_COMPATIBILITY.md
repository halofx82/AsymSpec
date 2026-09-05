# vLLM compatibility

This checkout targets exactly **vLLM 0.28.0**. Its eight-file deployment
payload is rebased on that release and must not be applied to other versions.
The original sibling checkout remains the 0.19.0 reference.

See [PORTING.md](PORTING.md) for validation status and the mandatory TP=4,
CPU-offloaded LongBench acceptance test. Deployment checks enforce both the
package version and the expected upstream source hashes. Repeated deployment
preserves the original backup; revert restores upstream files and removes
newly installed AsymSpec modules.

The port uses the V1 GPU runner. AsymSpec selects synchronous scheduling,
disables prefix caching, and preserves the original optional cross-family
mapping instead of enabling upstream automatic heterogeneous mapping. Other
speculative methods retain their upstream dispatch.
