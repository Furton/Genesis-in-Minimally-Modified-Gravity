(* Headless evaluation of 'Analytic fit.nb' with wolframscript (no front end needed).

   usage (from the repository root):  wolframscript -file scripts/evaluate_analytic_nb.wls

   Imports the notebook, evaluates its Input cells in order against genesis_scan_table.csv
   in the repository root, and prints the sample-comparison rows (fixed parameter
   identities of the paper's Table 1). Compare with data/fit_sample_table.csv. *)

root = DirectoryName[DirectoryName[$InputFileName]];
SetDirectory[root];
nb = Import["Analytic fit.nb", "Notebook"];
inputs = Cases[nb, Cell[BoxData[b_], "Input", ___] :> b, Infinity];
Print["input cells: ", Length[inputs]];
(* the notebook locates the CSV via NotebookDirectory[], which is unavailable without a front end *)
Unprotect[NotebookDirectory]; NotebookDirectory[] = root; Protect[NotebookDirectory];
Do[
  held = ToExpression[inputs[[i]], StandardForm, Hold];
  If[held === $Failed, Print["cell ", i, ": failed to parse"]; Exit[1], ReleaseHold[held]],
  {i, Length[inputs]}];
Print["rows imported: ", Length[r], "   coefficients (a, b, q): ", {a, b, q}];
Do[Print[Row[Map[InputForm, First[row]], "  "]], {row, results}];
