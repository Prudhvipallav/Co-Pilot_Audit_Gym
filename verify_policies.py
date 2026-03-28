from app.tasks import load_task
from app.policies import get_violations

for tid in [1, 2, 3]:
    task = load_task(tid)
    gt = task['ground_truth_violations']
    found = get_violations(task['artifacts'])
    hit = [v for v in gt if v in found]
    miss = [v for v in gt if v not in found]
    fp = [v for v in found if v not in gt]
    print(f"Task {tid} ({task['feature_name']}):")
    print(f"  GT:    {gt}")
    print(f"  Found: {found}")
    print(f"  Hit:   {hit}  Miss: {miss}  FP: {fp}")
    print()
