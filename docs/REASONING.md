# Problem-solving answers

## 1. If 30% of predictions are wrong

First I would ask where the 30% came from. If it came from an evaluation report, I would check the dataset, label join, sample count, model version, and pipeline digest. If it came from analyst corrections, I would check how those tiles were selected and whether the labels were reviewed consistently. Then I would break the errors down by class, source, and time. A single 70% accuracy figure hides whether the mistakes are mostly harmless or whether most real industrial tiles are called residential.

I would inspect the confusion matrix and a few wrong tiles before changing the model. I would check for bad labels, split leakage, preprocessing differences, mixed tiles, and distribution shift. I would compare the service with a simple baseline and the analyst's current workflow. Then I would ask which mistakes cost the most and how many tiles an analyst can review. If a score threshold gives useful coverage with an acceptable error rate on validation data, the model could still help sort tiles for review. I would report accepted coverage and accepted error together, and keep human corrections separate from model predictions. If costly errors remain high, the model is not suitable for that decision.

## 2. A month offline and unattended

I would first ask how the problem was noticed. If a local dashboard shows a gap, I would check its last-update time and whether it reads the same database as the API. If an operator found it after returning, I would check persisted logs and counters rather than assume the missing remote alert means the service was healthy.

I would inspect process restarts, failed requests, inference latency, disk space, database integrity, and missing input files. I would run fixed canary tiles through the deployed checkpoint and preprocessing, then compare the output and digests with the recorded baseline. I would sample recent tiles for analyst labels and compare errors by class and acquisition source. A healthy endpoint and unchanged canaries show that the pipeline still runs. They do not prove that new imagery remains accurate. That needs fresh labels or a review process.

## 3. Ingestion succeeds but stored results look wrong

I would first pin down how the mismatch was found. If an analyst says the displayed class is wrong, I would capture that result ID and the tile they reviewed. If the API response differs from the history page or a database query, I would compare those exact views before looking at model quality. I would preserve the original row and image before trying any repair.

For that result, I would compare the submitted bytes and hash, saved image, model version, pipeline digest, ordered class map, scores, and timestamp. I would rerun the saved bytes through the exact checkpoint and preprocessing. If the replay differs, I would inspect model replacement, transforms, and runtime version. If it matches, I would compare the inference response with the SQLite row and API response. I would check class-index mapping, request-to-image association, SQL bindings, and serialization. Then I would check whether the tile or its ground-truth label is itself wrong. I would trace a few more affected rows before changing the model. A class-order bug can make every stored label wrong while uploads and latency still look healthy.

## 4. Weakest point

The weakest claim is that performance on a small RGB EuroSAT subset will carry over to GalaxEye's operational imagery. The data have no location, season, or source-scene IDs, and they are not SAR or multisensor inputs. A new sensor, geography, season, or cloudy scene could break accuracy before the API fails. I would treat these predictions as advisory, preserve originals and pipeline identity, ask for representative labeled examples from the deployment setting, and monitor class-level errors over time.

I would also watch disk use and failed-ingestion logs. The API writes the image file before it calls inference. If inference fails, the request returns an error but leaves that file behind. Startup cleanup removes old files that have no database row, but it does not run continuously. This is a code-level failure path that needs a focused reproduction and cleanup policy before calling it a confirmed operational incident.

## What I checked so far

- The local API and inference service report healthy, and the API lists all three model versions.
- All three checkpoint hashes match their manifests and evaluation reports. The split digests also match.
- The SQLite integrity check passes. The database has seven saved predictions, no duplicate image-and-pipeline results, no images without predictions, and no missing image files.
- I replayed the seven saved images through their recorded model versions. Their hashes, predicted classes, and scores match the stored rows exactly. This rules out a mismatch in those seven examples only.
- The supplied evaluation results do not show a 30% error rate. V1 has 15 errors out of 210, v2 has 8, and v3 has 9. These are small fixed-set measurements, not proof of operational accuracy.
- V1's largest evaluation confusions are River to Highway (6 tiles) and AnnualCrop to River (4 tiles). V2 predicts the correct class for 10 of v1's 15 errors. V3 corrects 12. This points first to model and tile error analysis, not a demonstrated persistence fault.
- The staged-image cleanup path above is the clearest code-level issue I found. I have not yet reproduced a failed inference request leaving an orphan file.
