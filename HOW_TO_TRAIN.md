STEP 2 — Collect Training Videos-------------------------------------------------

Record videos with:

chairs
tables
people
movement
turning
obstacles
empty rooms
close/far objects

Good dataset = MANY environments.

Examples:

videos/
    room1.mp4
    hallway.mp4
    office.mp4
    kitchen.mp4

Aim for:

20–60 minutes total video
multiple lighting conditions
moving camera
different object distances
STEP 3 — Generate Initial Labels Automatically-------------------------------------------------

Use your autolabel mode:

python video_processor.py autolabel --video videos/room1.mp4 --output data/raw/room1.csv



STEP 4 — Manually Correct Labels (IMPORTANT)----------------------------------------------------------------

Auto-labeling is NEVER enough.

Now manually refine data:

python video_processor.py label --video videos/room1.mp4 --session room1_fixed

Controls:

Key	Action
A	avoid person
C	move to chair
T	check table
E	explore
SPACE	pause
ESC	quit

This creates high-quality supervised data.

This is the MOST IMPORTANT step.

Good labels = good AI.

STEP 5 — Merge CSV Files----------------------------------------------------------------------------
python merge.py

STEP 6 — Split Train / Validation / Test------------------------------------------------------------------
python split_dataset.py

STEP 7 — Train the Model-----------------------------------------------------------------------------

Now train:

python train_reasoning.py --train data/processed/train.csv --val data/processed/val.csv --test data/processed/test.csv --epochs 40 --batch-size 64 --lr 0.001

STEP 8 — What Happens During Training--------------------------------------------------------------

You’ll see:

Epoch 1/40 loss=1.22 train_acc=0.61 val_acc=0.58
Epoch 2/40 loss=0.91 train_acc=0.74 val_acc=0.70
Epoch 3/40 loss=0.62 train_acc=0.83 val_acc=0.79

The model automatically saves:

models/reasoning_model.pt

And generates:

reports/metrics.json
reports/confusion_matrix.png

STEP 9 — Use the Trained AI---------------------------------------------------------------------

Now run the main system:

python main.py

Or on video:

python main.py --video videos/test.mp4

The AI will now use:

use_model=True

instead of only rule-based logic.

STEP 10 — Improve Accuracy

The biggest improvements come from:

Improvement	Effect
more videos	HUGE
more manual labels	HUGE
balanced classes	HUGE
LSTM sequence model	VERY HUGE
trajectory features	HUGE
depth estimation	MASSIVE
object velocity	MASSIVE
semantic maps	MASSIVE
STEP 11 — Recommended Next Upgrade (VERY IMPORTANT)

Right now your AI sees:

single frame → action

Real robots use:

sequence of frames → action

That’s why your SequenceDataset matters.

Next step:

Replace MLP with LSTM

This will massively improve:

motion understanding
temporal reasoning
prediction stability
navigation quality


STEP 12 — Expected Accuracy

Approximate:

System	Accuracy
rule-based	55–65%
MLP	70–85%
LSTM	85–93%
LSTM + depth	90–97%
STEP 13 — Best Dataset Strategy

The BEST dataset:

70% auto-label
30% manual correction

This scales fast while keeping quality high.

STEP 14 — If Training Fails

Common fixes:

CUDA out of memory

Lower batch size:

--batch-size 16
Overfitting

Symptoms:

train_acc=99%
val_acc=60%

Fixes:

more data
stronger dropout
early stopping
data augmentation
Low accuracy

Usually caused by:

bad labels
insufficient diversity
too few videos
class imbalance
STEP 15 — REAL Robotics Upgrade Path

Your current system is already becoming a real robotics AI stack.

Next realistic upgrades:

YOLOv8 tracking
LSTM reasoning
Monocular depth estimation
SLAM mapping
Path planning
Reinforcement learning
Semantic memory
Goal planner
Autonomous exploration
Multi-agent reasoning

That becomes an actual autonomous navigation AI.