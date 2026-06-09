# Running the Project on Ubuntu 24.04 (UTM VM) with VSCode
### Step-by-step guide — no prior experience needed

---

## Overview

Your Ubuntu 24.04 VM in UTM already has VSCode installed. You just need to:
1. Install the C++ compiler, CMake, and OpenCV (one command)
2. Install two VSCode extensions
3. Copy the project into the VM
4. Build and run

Total setup time: about 10–15 minutes.

---

## PART 1 — Share the project folder between Mac and VM

You need to get the `Cvision` project folder (which is on your Mac) into the Ubuntu VM.

**Option A — UTM Shared Folder (easiest)**

1. In UTM, click your Ubuntu VM → click the **settings gear** icon (top-right of the VM window)
2. Go to **Sharing** tab
3. Click **+** and select the folder on your Mac that contains `Cvision/`
4. Start the VM
5. Inside Ubuntu, the shared folder appears at `/media/share/` or you can mount it:
   ```bash
   # Inside Ubuntu terminal:
   ls /media/share/
   ```
6. Copy it to your home folder:
   ```bash
   cp -r /media/share/Cvision ~/Cvision
   ```

**Option B — USB drive**

Copy `Cvision/` to a USB drive, plug it into your Mac (and it will be shared with the VM automatically if UTM USB passthrough is enabled), then copy it into Ubuntu.

**Option C — Type the path directly**

If your dataset is also on the Mac, copy it to the VM via shared folder the same way.

> **Note:** After copying, all further steps happen **inside the Ubuntu VM**.

---

## PART 2 — Install build tools and OpenCV

Open a Terminal inside Ubuntu VM.

**Step 2.1 — Update the package list**

```bash
sudo apt update
```

Type your Ubuntu password when prompted.

**Step 2.2 — Install everything in one command**

```bash
sudo apt install -y build-essential cmake libopencv-dev libopencv-contrib-dev pkg-config
```

This installs:
- `build-essential` — the C++ compiler (g++)
- `cmake` — the build system
- `libopencv-dev` — the OpenCV computer vision library
- `pkg-config` — helps CMake find OpenCV

The download is about 200 MB and takes 3–5 minutes.

**Step 2.3 — Verify everything installed**

```bash
g++ --version
cmake --version
pkg-config --modversion opencv4
```

You should see version numbers for all three. ✓

Example output:
```
g++ (Ubuntu 13.2.0) 13.2.0
cmake version 3.28.3
4.8.0
```

---

## PART 3 — Install VSCode Extensions

Open **VSCode** inside Ubuntu (it should be in your Applications menu or Dock).

**Step 3.1** — Click the **Extensions icon** in the left sidebar (four squares, one slightly apart).

**Step 3.2** — In the search box, type `C/C++` and install **C/C++** by *Microsoft*.

**Step 3.3** — Search for `CMake Tools` and install **CMake Tools** by *Microsoft*.

**Step 3.4** — If VSCode asks you to reload, click **Reload** or restart VSCode.

---

## PART 4 — Open the project in VSCode

**Step 4.1** — In VSCode: `File → Open Folder…`

**Step 4.2** — Navigate to `~/Cvision` (or wherever you copied it) and click **Open**.

You will see the project on the left:
```
Cvision/
├── include/        ← header files (.hpp)
├── src/            ← source files (.cpp)
├── CMakeLists.txt
├── dataset_check.sh
└── ...
```

---

## PART 5 — Configure CMake (one-time setup)

**Step 5.1** — VSCode should automatically detect `CMakeLists.txt` and show a notification saying **"Would you like to configure this project?"** — click **Yes**.

> If you don't see the notification, press `Ctrl+Shift+P`, type `CMake: Configure`, press `Enter`.

**Step 5.2 — Select a Kit**

A dropdown will appear at the top of the screen. You will see options like:
- `GCC 13.2.0 x86_64-linux-gnu`

Select the **GCC** option. If there is only one, select it.

**Step 5.3 — Choose build type**

If asked, select **Release**.

The CMake output panel at the bottom will show:
```
[cmake] -- OpenCV  : 4.8.0
[cmake] -- Build   : Release
[cmake] -- Configuring done
[cmake] -- Build files have been written to: .../Cvision/build
```

This means CMake found the compiler and OpenCV correctly. ✓

> **If CMake says "Could not find OpenCV"**, run this in the terminal and try again:
> ```bash
> export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
> ```
> Then `Ctrl+Shift+P` → `CMake: Configure`.

---

## PART 6 — Build the project

**Step 6.1** — At the bottom of the VSCode window, look for a **Build** button (the gear ⚙ icon, or text saying `[all]`).

Click it.

**Alternatively:** Press `Ctrl+Shift+P` → type `CMake: Build` → press `Enter`.

The terminal panel will show the compilation. After 30–60 seconds you will see:
```
[100%] Linking CXX executable har
→ Binary copied to project root: ./har
[100%] Built target har
```

The compiled program `har` is now in your `Cvision/` folder. ✓

---

## PART 7 — Verify your dataset

Open the VSCode integrated terminal: `Terminal → New Terminal` in the menu.

Make the checker script executable and run it:

```bash
chmod +x dataset_check.sh
./dataset_check.sh /path/to/your/dataset
```

Replace `/path/to/your/dataset` with the actual location.

> **Tip:** Type `./dataset_check.sh ` (space at end) then drag your dataset folder from the Ubuntu file manager into the terminal to auto-fill the path.

The script will check:
- That each sequence folder has image files (`.jpg` / `.png`)
- That each folder has a `gt.txt` ground truth file
- That class IDs in `gt.txt` are valid (1–6)
- That you have 72 total sequences (12 per class)

If everything is green, you are ready to run. ✓

---

## PART 8 — Run the program

In the VSCode terminal:

```bash
./har /path/to/your/dataset --loocv --visualize
```

**What happens step by step:**

1. **Feature extraction** (~1–3 min): loads all 72 sequences, computes background models, tracks human bounding boxes, extracts 29 motion features per sequence
2. **Classification** (~5–10 min): runs Leave-One-Out Cross Validation — trains one SVM per sequence (72 total)
3. **Evaluation**: prints mIoU, accuracy, F1, confusion matrix in the terminal
4. **Output files**:
   - `eval_report.csv` — CSV report (open in LibreOffice Calc or similar)
   - `output_frames/` — annotated images, one per sequence

---

## Expected output

```
╔══════════════════════════════════════════════╗
║  Human Action Recognition  –  KTH Dataset   ║
╚══════════════════════════════════════════════╝

[STEP 1/3] Feature extraction from 72 sequences...
  [ 1/72] boxing_d1_seq1 (class 4)  IoU@20=0.821
  ...

[STEP 2/3] Classification...
  Leave-One-Out CV (72 iterations)...
  LOOCV 12/72  acc so far = 83.33%
  LOOCV 24/72  acc so far = 85.42%
  ...
  Final LOOCV accuracy: 87.50%

[STEP 3/3] Evaluation...

  mIoU (frame 20)  : 0.7341
  Global Accuracy  : 87.50 %

  Per-class F1 Scores:
    walking       : 0.9231
    jogging       : 0.8000
    running       : 0.8421
    boxing        : 0.9167
    handwaving    : 0.8696
    handclapping  : 0.8571
    Macro F1      : 0.8681

  Confusion Matrix  (row = Ground Truth,  col = Predicted)
                walking jogging running  boxing handwa handcl
  walking            12       0       0       0      0      0
  jogging             0      10       2       0      0      0
  running             0       1      11       0      0      0
  boxing              0       0       0      12      0      0
  handwaving          0       0       0       0     11      1
  handclapping        0       0       0       0      1     11
```

---

## Viewing annotated output images

The images are in `~/Cvision/output_frames/`. Open the **Files** app in Ubuntu, navigate there, and double-click any image to view it.

Each image shows:
- **Red box** = system's predicted bounding box at frame 20
- **Green box** = ground truth bounding box
- **Label** (top-right corner) = predicted action class name

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `sudo: command not found` | Type `su -` first, enter root password, then run the apt commands. |
| `apt: package not found` | Run `sudo apt update` first, then retry. |
| CMake: `Could not find OpenCV` | Run `sudo apt install libopencv-dev` and retry `CMake: Configure`. |
| `./har: Permission denied` | Run `chmod +x har` then try again. |
| Dataset not found | Run `./dataset_check.sh /path` and check the output. |
| CMake Tools extension grayed out | Make sure you opened a **folder** (not a file): `File → Open Folder`. |
| Shared folder not visible in VM | In UTM VM settings → Sharing, make sure the shared folder is enabled and the VM is restarted. |
| Build shows `filesystem` errors | Run `sudo apt install g++-13` to ensure you have a modern compiler. |

---

## Quick-reference commands

```bash
# Check dataset before running
./dataset_check.sh /path/to/dataset

# Full evaluation with annotated output images
./har /path/to/dataset --loocv --visualize

# Save the trained model (avoids re-training next time)
./har /path/to/dataset --loocv --save-model model.svm

# Load saved model (runs much faster)
./har /path/to/dataset --load-model model.svm --visualize

# Custom output folder and CSV file name
./har /path/to/dataset --loocv \
      --visualize --output my_results/ --report report.csv

# Show all available options
./har --help
```

---

## File structure reminder

```
Cvision/
├── include/                ← all .hpp header files (do not edit unless you know C++)
│   ├── types.hpp
│   ├── dataset_loader.hpp
│   ├── detector.hpp
│   ├── feature_extractor.hpp
│   ├── action_classifier.hpp
│   └── evaluator.hpp
├── src/                    ← all .cpp source files
│   ├── main.cpp
│   ├── dataset_loader.cpp
│   ├── detector.cpp
│   ├── feature_extractor.cpp
│   ├── action_classifier.cpp
│   └── evaluator.cpp
├── CMakeLists.txt          ← build instructions (do not edit)
├── dataset_check.sh        ← dataset validator script
├── setup_macos.sh          ← automated build for macOS
├── setup_linux.sh          ← automated build for Linux
├── SETUP_MACOS_VSCODE.md   ← macOS VSCode instructions
└── SETUP_UBUNTU_VSCODE.md  ← this file
```
