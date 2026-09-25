# Local model results

All three models trained on the same 735 candidate images and selected checkpoints using 105 validation images. The 210-image internal test and GalaxEye's separate 210-image evaluation set were examined after checkpoint and review policy selection. Every class has 105 train, 15 validation, 30 internal test, and 30 supplied evaluation examples.

| Version | Architecture | Validation accuracy | Internal test accuracy / macro-F1 | Supplied evaluation accuracy / macro-F1 | Internal-test CPU p50 / p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | YOLO26n classification, 128 px | 95.2% | 92.4% / 92.2% | 92.9% / 92.8% | 3.6 / 3.8 ms |
| v2 | MobileNetV3-Small, seven-class head, 224 px | 95.2% | 94.3% / 94.1% | 96.2% / 96.2% | 104.9 / 109.0 ms |
| v3 | YOLO26s classification, 128 px | 98.1% | 95.2% / 95.2% | 95.7% / 95.7% | 6.1 / 6.9 ms |

These figures are from the per-version `metrics.json` files, produced locally on this Apple Silicon Mac with PyTorch CPU inference and four CPU threads. They include Python preprocessing and model prediction per tile but exclude the HTTP request and SQLite commit. First-load time and total service memory are separate costs. The evaluation sets are small, so these results should not be treated as a general model ranking.

The review threshold for each version was chosen on validation using a preset grid from 0.50 to 0.95, maximizing accepted coverage while keeping observed error among accepted predictions at or below 10%. All three selected 0.50. On validation, v1 accepted 101/105, v2 accepted 105/105, and v3 accepted 104/105. These thresholds are a limitation of the small set and softmax score, not evidence of calibrated confidence.

The candidate and supplied evaluation images have no exact pixel duplicates, but no geography/source-scene identifiers are provided. Adjacent-scene or source overlap cannot be ruled out. These RGB results do not establish performance on SAR or multispectral imagery. The saved per-image CSVs and confusion matrices support error review without silently changing the reported checkpoint or split.
