# 🖥️ AWS Training Guide — GovernanceReview-Gym v3

> **Recommended Instance**: g5.xlarge (A10G 24GB) — Best use of $100 credits.
> **Budget Instance**: g4dn.xlarge (T4 16GB) — Slower, but fits in 8GB.

---

## Step 0 — Best Use of Your $100 Credits

Since you have $100 in credits, **DO NOT** use the T4 (`g4dn.xlarge`) unless you are restricted by quotas. 
The **A10G (`g5.xlarge`)** is 3-4x faster and only costs ~$1.00/hr (or ~$0.35 on Spot). 

| Instance | GPU | VRAM | Training Time | Total Cost |
|---|---|---|---|---|
| **g5.xlarge** | A10G | 24GB | **~4 hours** | ~$4.00 |
| **g4dn.xlargVe** | T4 | 16GB | **~12 hours** | ~$6.00 |

---

## Step 1 — Find Latest Deep Learning AMI (Run in AWS CloudShell)

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2*Ubuntu 22.04*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
  --output text
```

**Expected:** `ami-09b44f092d9d77f01   Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7...`

Save the AMI ID shown.

---

## Step 2 — Create Key Pair

```bash
aws ec2 create-key-pair \
  --key-name governance-gym-key \
  --query 'KeyMaterial' \
  --output text > governance-gym-key.pem
chmod 400 governance-gym-key.pem
```

---

## Step 3 — Launch Instance

Replace `YOUR_AMI_ID` with the ID from Step 1:

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id YOUR_AMI_ID \
  --instance-type g4dn.xlarge \
  --key-name governance-gym-key \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=governance-gym}]' \
  --query 'Instances[0].InstanceId' \
  --output text)
echo "Instance: $INSTANCE_ID"
```

**Expected:** `Instance: i-0abc123def456789`

---

## Step 4 — Wait + Get IP

```bash
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)
echo "IP: $PUBLIC_IP"
```

**Expected:** `IP: 54.123.456.789`

---

## Step 5 — SSH Into Instance

```bash
# Download key to local machine first, then:
ssh -i governance-gym-key.pem ubuntu@$PUBLIC_IP
```

---

## Step 6 — Setup (Run on EC2)

```bash
# Activate PyTorch env (already installed on DL AMI)
source activate pytorch

# Clone repo
git clone https://huggingface.co/spaces/Prudhvi06/Governance_review
cd Governance_review

# Install dependencies
pip install -q -r requirements.txt
pip install -q trl accelerate peft datasets

# Verify GPU
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

**Expected:** `GPU: Tesla T4 | VRAM: 15.8 GB`

---

## Step 7 — Start FastAPI Server (Background)

```bash
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
sleep 5
curl http://localhost:8000/health
```

**Expected:** `{"status":"healthy","version":"3.0.0"}`

---

## Step 8 — Run Training

```bash
# Run training with GRPO (takes ~4-12 hrs depending on instance)
python3 training/run_training.py \
  --total_steps 300 \
  --checkpoint_every 50 \
  --lora_r 16
```

**Note**: `run_training.py` is pre-configured to handle 8GB VRAM (T4) or 24GB VRAM (A10G) automatically by detecting your GPU's capability and adjusting `bf16`/`fp16` and batch sizes.

**Expected first 10 steps:**
```
[HH:MM:SS] GovernanceReview-Gym v3 — GRPO Training v5
[HH:MM:SS] GPU: Tesla T4 | Trainable: 2,162,688 / 496,195,456 (0.4%)
[HH:MM:SS] Dataset: 2000 prompts (50% easy, 50% hard)

{'rewards/reward_fn/mean': '-2.5', 'reward_std': '1.8', 'entropy': '2.1'}  ← HEALTHY
{'rewards/reward_fn/mean': '-1.8', 'reward_std': '1.6', 'entropy': '2.0'}
...
```

**Healthy signals to watch:**

| Metric | Good | Bad → Stop |
|---|---|---|
| `reward_std` | **>1.0** | <0.3 by step 10 |
| `entropy` | **>1.5** | <0.5 |
| `rewards/mean` | **Negative first, climbing** | Flat at -3.0 |

---

## Step 9 — Save Checkpoint to S3

```bash
aws s3 cp checkpoints/ s3://YOUR-BUCKET/governance-gym/checkpoints/ --recursive
echo "✅ Saved"
```

---

## Step 10 — Evaluate After Training

```bash
# Start server if not running
curl http://localhost:8000/health || \
  nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Run all 3 tasks
for task in 1 2 3; do
  python3 agents/orchestrator.py --mode static --task_id $task
done
```

**Expected improvement over base Qwen:**
```
Task 1: Grader 1.00 | Caught: ['PII-001'] | Fallbacks: <5
Task 2: Grader 0.85+ | Caught: ['RETENTION-005', 'ESCALATION-003']
Task 3: Grader 0.70+ | Caught: ['DOMAIN-004', 'TRAINING-006']
```

---

## Step 11 — Stop Instance (IMPORTANT — saves money)

```bash
# From CloudShell:
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "✅ Instance stopped (not terminated — data preserved)"
```

> ⚠️ **Always stop instance when done.** Stopped instance = no compute charges. Storage ~₹8/day.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `InvalidAMIID.NotFound` | Re-run Step 1 — AMI IDs are region-specific |
| SSH timeout | Add security group rule: port 22 from your IP |
| OOM on T4 | Use `--batch_size 2 --lora_r 8` |
| Server not responding | `cat server.log` to debug |
| Resume after stop | Re-run Step 8 with `--resume_from checkpoints/step_150` |
