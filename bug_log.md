# Bug Log

## Known Issues

### Version 1.0.0

#### High Priority
- [ ] FAISS index corruption under concurrent access - implement read-write locks
- [ ] Memory leak in PatchCore batch inference - investigate tensor cleanup

#### Medium Priority
- [ ] Knowledge graph query timeouts on large graphs (>100k nodes)
- [ ] LLM API rate limiting causes intermittent 429 errors
- [ ] XAI heatmap artifacts on boundary patches

#### Low Priority
- [ ] Windows path handling in data pipeline
- [ ] Inconsistent logging timestamps in distributed setup
- [ ] Docker build optimization for faster iterations

## Resolved Issues

### Version 0.9.0
- ✓ Fixed numerical instability in patch normalization
- ✓ Corrected FAISS serialization for multi-GPU setups
- ✓ Improved knowledge graph construction memory usage

## Test Coverage

- Unit Tests: 75%
- Integration Tests: 60%
- E2E Tests: 40%

## To Do

- [ ] Add distributed inference support
- [ ] Implement federated learning capability
- [ ] Add real-time performance monitoring dashboard
- [ ] Create mobile inference client
- [ ] Optimize FAISS index structure for faster queries
