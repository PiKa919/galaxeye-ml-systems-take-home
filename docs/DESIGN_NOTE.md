# Rough design notes

## What I checked in the data

| Check                      | Result                                                               |
| -------------------------- | -------------------------------------------------------------------- |
| Labeled candidate images   | 1,050, with 150 per class                                            |
| Separate evaluation images | 210, with 30 per class                                               |
| Classes                    | AnnualCrop, Forest, Highway, Industrial, Residential, River, SeaLake |
| Image format               | Every image decodes as a 64×64 RGB PNG                              |
| Exact pixel duplicates     | None found                                                           |
| Evaluation labels          | Every label maps to an image; no duplicate label rows                |
| Geographic metadata        | No coordinates, capture times, or other useful location metadata     |

- The archive identifies the images as a EuroSAT subset.
- These are optical RGB tiles. The assignment does not provide SAR bands or multispectral inputs.
- Without coordinates or resolution metadata, I cannot answer geographic queries or calculate forest area from these tiles.
- I split only the labeled candidate images. The separate 210-image evaluation set stayed out of training and model selection.

## Architecture options I considered

| Option                       | How it works                                                                     | Strength                                                        | Main drawback                                                 |
| ---------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| A. Synchronous local service | Upload, run CPU inference, commit to SQLite, return the result                   | Smallest complete system and easy to explain                    | A slow prediction holds up the request; bursts wait in line   |
| B. Durable local queue       | Save each job, let a worker classify it, and let the analyst poll for the result | A restart or client disconnect does not lose accepted work      | Needs job states, retry rules, leases, and recovery           |
| C. Separate services         | API and inference service, PostgreSQL, and image storage                         | Separates model processes and can scale each part independently | More setup and more failure handling than this exercise needs |

- I used separate local API and inference processes, with SQLite and local image files. This keeps the model process separate without adding PostgreSQL.
- The API validates a PNG, hashes it, and writes it under a hash-based path. It sends the bytes to the inference process on loopback.
- The inference process keeps the three CPU models loaded and returns seven ordered scores with the model digest.
- The API stores the image path and prediction in SQLite. It returns success only after the database commit.
- File storage and a SQLite commit cannot happen in one transaction. A crash can leave an image without a database row, so startup removes old orphan files. The database and image directory need a joint backup.
- A single-process service would be simpler for this small workload. A durable queue would be my next change if uploads must survive client disconnects or arrive in bursts.

## Model notes

- I trained YOLO26n classification as v1, MobileNetV3-Small as v2, and YOLO26s classification as v3.
- Each model used the same stratified 70/20/10 split: 735 training, 210 internal test, and 105 validation images.
- The separate evaluation set remained untouched until the models and review thresholds were selected.
- MobileNetV3-Small is a reasonable CPU baseline. Torchvision lists about 2.54 million parameters and a 9.8 MB pretrained weight file. That is the original weight-file size, not runtime memory. I would measure memory and latency on the target machine.
- The strongest softmax score is not calibrated confidence. I use a validation-selected threshold to flag tiles for review, but the score cannot prove a prediction is correct.
- The current data are small and lack source-scene identifiers. They cannot establish performance across geography, seasons, sensors, or weather.

## Questions I would ask GalaxEye

- How do you define an independent evaluation unit across sensors and acquisitions, and how do delayed analyst labels feed drift checks without mixing neighboring tiles from one scene across training and evaluation?
- What is the promotion and rollback process for models on isolated hardware, including signed offline bundles, resource limits, and the point at which new analyst labels are sufficient to approve an update?
