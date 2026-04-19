from collections import Counter

# ── Model priority weights for tie-breaking ────────────────────────────────────
# Used ONLY when two or more classes receive equal votes.
# Higher weight = higher confidence model for small tabular datasets.
# Ranking rationale:
#   SVM  (5) — best margin-based generalisation on small, imbalanced data
#   MLP  (4) — learns non-linear boundaries; robust with scaled features
#   KNN  (3) — solid baseline; sensitive to local structure
#   LR   (2) — linear boundaries; reliable but least expressive here
#   NB   (1) — fast but assumes feature independence (weakest assumption here)
MODEL_WEIGHTS = {
    "SVM"       : 5,
    "MLP"       : 4,
    "KNN"       : 3,
    "LR"        : 2,
    "NaiveBayes": 1,
}

def vote(predictions: dict) -> str:
    """
    Accept a dict of {model_name: predicted_class}, print a voting breakdown,
    and return the winning class.

    Tie-breaking: when two or more classes share the highest vote count,
    the winner is decided by summing the MODEL_WEIGHTS of every model that
    voted for each tied class. The class whose backers carry the most total
    weight wins. This is deterministic and justified by model quality.
    """

    answers     = list(predictions.values())
    vote_count  = Counter(answers)

    print("\n" + "-" * 50)
    print("         Voting Breakdown")
    print("-" * 50)

    for plant_type, count in vote_count.most_common():
        bar = "█" * count
        print(f"  {plant_type:<15} {bar}  ({count} vote(s))")

    print("-" * 50)

    top_count   = vote_count.most_common(1)[0][1]
    tied_winners = [cls for cls, cnt in vote_count.items() if cnt == top_count]

    if len(tied_winners) == 1:
        winner       = tied_winners[0]
        winner_votes = top_count
        print(f"\n  Winner: {winner} with {winner_votes} out of 5 votes")
        return winner

    # ── Tie detected — resolve by model weights ────────────────────────────────
    print(f"\n  Tie detected between: {tied_winners}")
    print("  Resolving by model confidence weights...")

    weight_totals = {cls: 0 for cls in tied_winners}
    for model_name, predicted_class in predictions.items():
        if predicted_class in weight_totals:
            weight_totals[predicted_class] += MODEL_WEIGHTS.get(model_name, 0)

    print("\n  Weight breakdown for tied classes:")
    for cls, w in sorted(weight_totals.items(), key=lambda x: -x[1]):
        backing_models = [m for m, p in predictions.items() if p == cls]
        print(f"    {cls:<15} total weight: {w}  (backed by: {', '.join(backing_models)})")

    winner = max(weight_totals, key=weight_totals.get)
    print(f"\n  Winner by weight: {winner} (weight {weight_totals[winner]})")
    return winner