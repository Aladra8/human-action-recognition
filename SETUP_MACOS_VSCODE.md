# Running the Project on macOS with VSCode
### Step-by-step guide — no prior experience needed

---

## What you will install

| Tool | What it is | How you get it |
|------|-----------|----------------|
| Xcode Command Line Tools | C++ compiler (not the Xcode app) | One Terminal command |
| Homebrew | macOS package manager | One Terminal command |
| OpenCV | Computer vision library | Via Homebrew |
| CMake | Build system | Via Homebrew |
| C/C++ extension | VSCode understands C++ | Inside VSCode |
| CMake Tools extension | Build button inside VSCode | Inside VSCode |

---

## PART 1 — Install the compiler (Xcode Command Line Tools)

> **Important:** You do NOT need to install the full Xcode app (it is 12 GB).
> You only need the lightweight "Command Line Tools" (about 500 MB).

**Step 1.1 — Open Terminal**

- Press `Command ⌘ + Space` to open Spotlight Search
- Type `Terminal` and press `Return`

**Step 1.2 — Run this command**

Copy and paste this exactly into Terminal, then press `Return`:

```
xcode-select --install
```

**Step 1.3 — Click "Install" in the popup**

A dialog box will appear asking if you want to install the tools.
Click **Install** (not "Get Xcode"). The download takes 5–15 minutes.

**Step 1.4 — Verify it worked**

After the download finishes, type this in Terminal and press `Return`:

```
clang++ --version
```

You should see something like:
```
Apple clang version 15.0.0 ...
```
If you see a version number, the compiler is installed. ✓

---

## PART 2 — Install Homebrew

Homebrew lets you install OpenCV and CMake with simple commands.

**Step 2.1 — Run the Homebrew installer**

In Terminal, paste this entire line and press `Return`:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your **Mac login password** — type it (you won't see the characters) and press `Return`. The install takes 2–5 minutes.

**Step 2.2 — For Apple Silicon Macs (M1/M2/M3/M4 chip) only**

At the end of the Homebrew install, you will see a message that says:

```
==> Next steps:
Run these commands in your terminal to add Homebrew to your PATH:
    eval "$(/opt/homebrew/bin/brew shellenv)"
```

Copy that `eval` line from YOUR terminal output and paste it, then press `Return`. Then run:

```
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
```

**Step 2.3 — Verify**

```
brew --version
```

Should show `Homebrew 4.x.x`. ✓

---

## PART 3 — Install OpenCV and CMake

In Terminal, run these two commands one at a time:

```
brew install cmake
```

Then (this one takes 5–10 minutes):

```
brew install opencv
```

**Verify both installed:**

```
cmake --version
pkg-config --modversion opencv4
```

You should see version numbers for both. ✓

---

## PART 4 — Install VSCode (if not already installed)

**Step 4.1** — Go to [https://code.visualstudio.com](https://code.visualstudio.com) and click **Download Mac**.

**Step 4.2** — Open the downloaded `.zip`, then drag `Visual Studio Code.app` into your **Applications** folder.

**Step 4.3** — Open VSCode from Applications.

---

## PART 5 — Install VSCode Extensions

You need two extensions. In VSCode:

**Step 5.1** — Click the **Extensions icon** in the left sidebar (it looks like four squares, one slightly detached).

**Step 5.2** — Search for `C/C++` and install **C/C++** by *Microsoft*.

**Step 5.3** — Search for `CMake Tools` and install **CMake Tools** by *Microsoft*.

**Step 5.4** — VSCode may ask you to reload — click **Reload** or **Restart** if prompted.

---

## PART 6 — Copy the project to your Mac and open it

**Step 6.1** — Copy the `Cvision` folder to somewhere convenient, for example your Desktop or your Documents folder.

**Step 6.2** — In VSCode, go to `File → Open Folder…`

**Step 6.3** — Navigate to the `Cvision` folder and click **Open**.

You will now see the project tree on the left:
```
Cvision/
├── include/
├── src/
├── CMakeLists.txt
├── setup_macos.sh
└── ...
```

---

## PART 7 — Configure the build (one-time setup)

**Step 7.1** — VSCode should automatically detect `CMakeLists.txt` and show a notification at the bottom or top asking to "Configure" or "Select a Kit". Click it.

> If you do NOT see this notification, press `Cmd+Shift+P`, type `CMake: Configure`, and press `Return`.

**Step 7.2 — Select a Kit**

A dropdown will appear at the top. You will see options like:
- `Clang 15.0.0 arm64-apple-darwin...`  ← choose this one on Apple Silicon
- `GCC 13 ...`

Select the one that shows **Clang** or **Apple Clang**. If you only see one option, select it.

**Step 7.3 — Choose build type**

If asked, choose **Release** (for better performance).

CMake Tools will run in the terminal at the bottom and show:
```
[cmake] -- OpenCV  : 4.x.x
[cmake] -- Build   : Release
[cmake] -- Configuring done
```

This means it found OpenCV successfully. ✓

> **If CMake says "Could not find OpenCV"** — run this in Terminal and then retry:
> ```
> export OpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
> ```
> Then in VSCode press `Cmd+Shift+P` → `CMake: Configure`.

---

## PART 8 — Build the project

**Step 8.1** — At the very bottom of the VSCode window, find the status bar. You will see a **Build** button (looks like a gear ⚙ or says `[all]`).

**Alternatively:** Press `Cmd+Shift+P`, type `CMake: Build`, press `Return`.

The terminal at the bottom will show the compilation. It takes about 30–60 seconds. At the end you should see:
```
[100%] Linking CXX executable har
→ Binary copied to project root: ./har
[100%] Built target har
```

**The file `har` now exists** in your `Cvision` folder. ✓

---

## PART 9 — Verify your dataset

**Step 9.1** — Open the VSCode terminal: go to `Terminal → New Terminal` in the menu.

**Step 9.2** — Run the dataset checker (replace the path with the actual location of your dataset):

```bash
chmod +x dataset_check.sh
./dataset_check.sh /path/to/your/dataset
```

> **Tip:** Instead of typing the path, type `./dataset_check.sh ` (with a space at the end), then drag your dataset folder from Finder into the Terminal. It will auto-fill the path.

The checker will tell you:
- How many sequences it found per class
- Whether the image files are present
- Whether the `gt.txt` files are correctly formatted
- Whether you are ready to run

---

## PART 10 — Run the program

In the VSCode terminal:

```bash
./har /path/to/your/dataset --loocv --visualize
```

Again, you can drag the dataset folder into Terminal to fill in the path.

**What happens:**
1. It loads all 72 sequences and extracts features (takes 1–3 minutes)
2. It runs Leave-One-Out Cross Validation (trains 72 SVMs — takes 3–8 minutes)
3. It prints the evaluation report (mIoU, accuracy, F1, confusion matrix)
4. It saves `eval_report.csv` in the `Cvision/` folder
5. It saves annotated images in `output_frames/`

---

## Expected terminal output

```
╔══════════════════════════════════════════════╗
║  Human Action Recognition  –  KTH Dataset   ║
╚══════════════════════════════════════════════╝

[STEP 1/3] Feature extraction from 72 sequences...
  [ 1/72] boxing_d1_seq1 (class 4)  IoU@20=0.821
  [ 2/72] boxing_d2_seq2 (class 4)  IoU@20=0.774
  ...

[STEP 2/3] Classification...
  Leave-One-Out CV (72 iterations)...
  LOOCV 12/72  acc so far = 83.33%
  LOOCV 24/72  acc so far = 85.42%
  ...
  Final LOOCV accuracy: 87.50%

[STEP 3/3] Evaluation...

╔══════════════════════════════════════════════╗
║       EVALUATION REPORT                      ║
╚══════════════════════════════════════════════╝
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
```

---

## Viewing the output images

The annotated images are saved in `Cvision/output_frames/`.
Open them in Finder — each image shows:
- **Red box** = system's predicted bounding box
- **Green box** = ground truth bounding box
- **Label** (top-right) = predicted action name

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `xcode-select: error: command line tools are already installed` | That's fine — it's already installed. Move to Part 2. |
| `brew: command not found` | Homebrew didn't install correctly. Re-run the install command from Part 2. Apple Silicon users: make sure you ran the `eval` command. |
| CMake: `Could not find OpenCV` | Run `export OpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4` in Terminal, then `CMake: Configure` in VSCode. |
| `./har: Permission denied` | Run `chmod +x har` then try again. |
| No sequences loaded | Run `./dataset_check.sh /your/dataset` and follow its suggestions. |
| CMake Tools extension doesn't show up | Reload VSCode: `Cmd+Shift+P` → `Developer: Reload Window`. |
| Build fails with `filesystem` errors | Make sure CMake selected the Clang compiler (not an old GCC). Redo Step 7.2. |

---

## Quick-reference commands

```bash
# Run full evaluation with visualisation
./har /path/to/dataset --loocv --visualize

# Save the trained SVM model for later re-use
./har /path/to/dataset --loocv --save-model model.svm

# Run with saved model (much faster, no re-training)
./har /path/to/dataset --load-model model.svm --visualize

# Custom output paths
./har /path/to/dataset --loocv \
      --visualize --output my_results/ --report my_report.csv

# Show all options
./har --help
```
