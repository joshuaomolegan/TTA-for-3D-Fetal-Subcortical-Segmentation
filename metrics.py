import torch
import surface_distance

def dice(ground_truth, prediction):
    # Calculate the dice coefficient of the ground truth and the prediction
    # dice coeff = 2 * |ground_truth ^ prediction| / (|ground_truth| + |prediction|)
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    intersection = torch.sum(ground_truth * prediction)
    cardinality = torch.sum(ground_truth) + torch.sum(prediction)
    smooth = 0.0001
    return (2. * intersection + smooth) / (cardinality + smooth)


def mulitclass_dice(ground_truth, prediction, numLabels, verbose=False):
    dice_score_dict = {}
    avg_dice_score = 0
    # Start from 1 to ignore background
    for label in range(1, numLabels+1):
        d = dice((ground_truth == label).float(), (prediction == label).float())
        dice_score_dict[f"label_{label}"] = d
        avg_dice_score += d

        if verbose:
            print(f"Dice for label {label}: {d.item()}")
    
    avg_dice_score /= numLabels
    dice_score_dict['avg'] = avg_dice_score

    return dice_score_dict

# Compute 95th percentile Hausdorff distance and normalised surface distance (surface dice)
def surface_metrics(ground_truth, prediction, numLabels, verbose=False):
    surface_dice_dict = {}
    haussdorf_dict = {}
    avg_surface_dice = 0
    avg_haussdorf = 0

    # Add batch dimension if not present
    if len(ground_truth.shape) == 3:
        ground_truth = ground_truth.unsqueeze(0)
    
    if len(prediction.shape) == 3:
        prediction = prediction.unsqueeze(0)

    # Start from 1 to ignore background
    for batchnum in range(prediction.shape[0]):
        for label in range(1, numLabels+1):

            pred = prediction[batchnum] == label
            gt = ground_truth[batchnum] == label

            # If both are empty, set surface dice to 1 and haussdorf to 0
            if not torch.any(pred) and not torch.any(gt):
                surface_dice_dict[f"label_{label}"] = 1.0
                haussdorf_dict[f"label_{label}"] = 0.0
                continue

            dist = surface_distance.compute_surface_distances(
                    pred.cpu().numpy(), gt.cpu().numpy(), 
                    spacing_mm=(0.6, 0.6, 0.6))

            haussdorf = surface_distance.compute_robust_hausdorff(dist, 95)
            surface_dice = surface_distance.compute_surface_dice_at_tolerance(dist, 0.6)
            
            surface_dice_dict[f"label_{label}"] = surface_dice_dict.get(f"label_{label}", 0.0) + surface_dice

            haussdorf_dict[f"label_{label}"] = haussdorf_dict.get(f"label_{label}", 0.0) + haussdorf

            if verbose:
                print(f"Batch num: {batchnum}. Surface Dice for label {label}: {surface_dice}")
                print(f"Batch num: {batchnum}. Haussdorf for label {label}: {haussdorf}")
    
    # Average over all batches
    for label in range(1, numLabels+1):
        surface_dice_dict[f"label_{label}"] /= prediction.shape[0]
        haussdorf_dict[f"label_{label}"] /= prediction.shape[0]

        avg_surface_dice += surface_dice_dict[f"label_{label}"]
        avg_haussdorf += haussdorf_dict[f"label_{label}"]
    
    # Calculate an average over all labels
    avg_surface_dice /= numLabels 
    surface_dice_dict['avg'] = avg_surface_dice

    avg_haussdorf /= numLabels
    haussdorf_dict['avg'] = avg_haussdorf

    return surface_dice_dict, haussdorf_dict