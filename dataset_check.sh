#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# dataset_check.sh
#
# Run this BEFORE building the project to verify your downloaded dataset
# is structured correctly and that the program will find everything.
#
# Usage:
#   chmod +x dataset_check.sh
#   ./dataset_check.sh /path/to/your/downloaded/dataset
# ──────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; BLU='\033[0;34m'; NC='\033[0m'

ok()   { echo -e "  ${GRN}✔${NC}  $*"; }
warn() { echo -e "  ${YEL}⚠${NC}  $*"; }
err()  { echo -e "  ${RED}✘${NC}  $*"; }
info() { echo -e "  ${BLU}→${NC}  $*"; }

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/dataset"
    exit 1
fi

DATASET="$1"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║       Dataset Structure Checker              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Checking: $DATASET"
echo ""

# ── 1. Does the path exist? ───────────────────────────────────────────────────
if [[ ! -d "$DATASET" ]]; then
    err "Path does not exist: $DATASET"
    echo ""
    echo "  Fix: make sure you typed the path correctly."
    echo "  Tip: drag the folder from Finder into the Terminal to auto-fill the path."
    exit 1
fi
ok "Dataset folder exists."

# ── 2. Detect layout ──────────────────────────────────────────────────────────
CLASS_NAMES=("boxing" "handclapping" "handwaving" "jogging" "running" "walking")
HAS_CLASS_DIRS=false
for cls in "${CLASS_NAMES[@]}"; do
    if [[ -d "$DATASET/$cls" ]]; then
        HAS_CLASS_DIRS=true
        break
    fi
done

if $HAS_CLASS_DIRS; then
    echo "  Layout detected: A  (root/class_name/sequence_folder/)"
else
    echo "  Layout detected: B  (root/sequence_folder/  ← class from gt.txt)"
fi
echo ""

# ── 3. Scan sequences ─────────────────────────────────────────────────────────
declare -A class_count
for c in 1 2 3 4 5 6; do class_count[$c]=0; done

total_seqs=0
total_issues=0

scan_seq_folder() {
    local seq_dir="$1"
    local cls_hint="$2"   # optional class hint from parent dir name

    # Count image files
    img_count=$(find "$seq_dir" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
    img_count=$(echo $img_count | tr -d ' ')

    # Find gt.txt
    gt_file=$(find "$seq_dir" -maxdepth 1 -name "*.txt" | head -1)

    seq_name=$(basename "$seq_dir")
    has_issue=false

    # Check image count
    if [[ "$img_count" -eq 0 ]]; then
        err "[$seq_name] No image files found!"
        has_issue=true
    elif [[ "$img_count" -lt 40 ]]; then
        warn "[$seq_name] Only $img_count frames (expected 40)"
    else
        ok "[$seq_name] $img_count frames"
    fi

    # Check gt.txt
    if [[ -z "$gt_file" ]]; then
        err "[$seq_name] No .txt ground truth file!"
        has_issue=true
    else
        # Parse gt.txt
        read -r cls xc yc w h < "$gt_file"
        if [[ -z "$cls" ]]; then
            err "[$seq_name] gt.txt is empty or unreadable"
            has_issue=true
        elif [[ "$cls" -lt 1 || "$cls" -gt 6 ]] 2>/dev/null; then
            err "[$seq_name] gt.txt class_id=$cls is out of range (must be 1-6)"
            has_issue=true
        else
            # Normalised or absolute?
            norm_check=$(echo "$xc $yc $w $h" | awk '{
                if ($1<=1.5 && $2<=1.5 && $3<=1.5 && $4<=1.5) print "normalised"
                else print "absolute"
            }')
            info "[$seq_name] gt: class=$cls  coords=$norm_check ($xc $yc $w $h)"
            class_count[$cls]=$((class_count[$cls]+1))
        fi
    fi

    if $has_issue; then total_issues=$((total_issues+1)); fi
    total_seqs=$((total_seqs+1))
}

echo "── Scanning sequences ───────────────────────────────────────"
if $HAS_CLASS_DIRS; then
    for cls in "${CLASS_NAMES[@]}"; do
        cls_dir="$DATASET/$cls"
        if [[ ! -d "$cls_dir" ]]; then
            warn "Class directory missing: $cls/"
            continue
        fi
        echo ""
        echo "  Class: $cls"
        for seq_dir in "$cls_dir"/*/; do
            [[ -d "$seq_dir" ]] && scan_seq_folder "$seq_dir" "$cls"
        done
    done
else
    for seq_dir in "$DATASET"/*/; do
        [[ -d "$seq_dir" ]] && scan_seq_folder "$seq_dir" ""
    done
fi

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "── Summary ──────────────────────────────────────────────────"
echo ""
printf "  %-16s %s\n" "Class" "Sequences found"
printf "  %-16s %s\n" "─────" "───────────────"
declare -A cls_name
cls_name[1]="walking"; cls_name[2]="jogging"; cls_name[3]="running"
cls_name[4]="boxing";  cls_name[5]="handwaving"; cls_name[6]="handclapping"
for c in 1 2 3 4 5 6; do
    cnt=${class_count[$c]}
    if [[ "$cnt" -eq 12 ]]; then
        printf "  %-16s ${GRN}%d${NC}\n" "${cls_name[$c]}" "$cnt"
    elif [[ "$cnt" -gt 0 ]]; then
        printf "  %-16s ${YEL}%d  (expected 12)${NC}\n" "${cls_name[$c]}" "$cnt"
    else
        printf "  %-16s ${RED}%d  ← MISSING${NC}\n" "${cls_name[$c]}" "$cnt"
    fi
done
echo ""
echo "  Total sequences : $total_seqs  (expected 72)"
echo "  Issues found    : $total_issues"
echo ""

if [[ "$total_seqs" -eq 72 && "$total_issues" -eq 0 ]]; then
    echo -e "  ${GRN}✔  Dataset looks PERFECT. You are ready to run the program.${NC}"
    echo ""
    echo "  Next step (after building):"
    echo "    ./har \"$DATASET\" --loocv --visualize"
elif [[ "$total_seqs" -gt 0 && "$total_issues" -eq 0 ]]; then
    echo -e "  ${YEL}⚠  Dataset found but sequence count is not 72.${NC}"
    echo "     The program will still run with whatever sequences are present."
else
    echo -e "  ${RED}✘  Issues detected above. Fix them before running the program.${NC}"
    echo ""
    echo "  Common fixes:"
    echo "    • Make sure each sequence folder contains image files (.jpg/.png)"
    echo "    • Make sure each sequence folder has exactly one .txt file"
    echo "    • gt.txt format must be:  <class_id> <x_center> <y_center> <width> <height>"
    echo "    • class_id must be 1=walking 2=jogging 3=running 4=boxing 5=handwaving 6=handclapping"
fi
echo ""
