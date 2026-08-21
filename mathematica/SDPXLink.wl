(* ::Package:: *)

(* SDPXLink — call the SDPX.jl semidefinite-programming solver from Mathematica.

   First-version transport: a command-line bridge. The problem is exported as
   JSON (schema v1, documented in docs/src/bridge-schema.md), Julia is invoked with
   RunProcess on bin/sdpx_solve.jl, and the JSON result is imported back.
   One process per solve, no shared state, arbitrary-precision numbers as
   strings. The documented upgrade path to a persistent Julia server or
   LibraryLink lives in the same schema document; the schema, not the
   transport, is the stable contract.

   Setup (once):
       julia --project=bin -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()"
   run from the SDPX.jl repository root.

   Usage:
       << SDPXLink`
       SDPXOptimize[c, A, C]                    (* min c.x  s.t.  Sum_i x_i A[[l,i]] - C[[l]] >= 0 *)
       SDPXOptimize[c, A, C, B, b, opts]        (* plus equality constraints  Transpose[B].x == b   *)

   `A` is a list over PSD blocks; `A[[l]]` is a list over variables of square
   matrices (dense lists or SparseArray). `C` is the list of constant matrices.
   Returns an Association, or a Failure whose message carries the solver's own
   error text. *)

BeginPackage["SDPXLink`"];

SDPXOptimize::usage =
  "SDPXOptimize[c, A, C] and SDPXOptimize[c, A, C, B, b] solve the SDP \
minimize c.x subject to Sum_i x_i A[[l,i]] - C[[l]] \[SucceedsEqual] 0 and \
Transpose[B].x == b using SDPX.jl through its command-line bridge. Options: \
\"Precision\" (\"Float64\", \"Float64x2\", \"Float64x4\", \"BigFloat\"), \
\"PrecisionBits\", \"Tolerance\", \"MaxIterations\", \"TimeLimit\", \
\"Verbosity\", \"ReturnMatrices\", \"JuliaExecutable\", \"SDPXDirectory\", \
\"KeepFiles\".";

SDPXOptimize::julia = "The Julia process failed (exit code `1`). Stderr: `2`";
SDPXOptimize::noresult = "The bridge produced no result file. Stderr: `1`";
SDPXOptimize::solver = "SDPX reported an error: `1`";
SDPXOptimize::setup =
  "Could not find `1`. Set \"SDPXDirectory\" to the SDPX.jl repository root \
and run the one-time setup documented in mathematica/README.md.";

Begin["`Private`"];

(* Captured at load time: $InputFileName names this .wl file only while Get is
   reading it. Resolving it lazily at call time -- the first version's mistake
   -- yields the *caller's* script path, or nothing in a notebook. *)
$packageRoot = If[StringQ[$InputFileName] && $InputFileName =!= "",
    DirectoryName[$InputFileName, 2], Directory[]];

(* ---- number rendering ---------------------------------------------------- *)

(* Machine numbers travel as plain JSON numbers; anything requesting more
   precision travels as a string, because JSON numbers are IEEE doubles in
   most parsers and silently round everything wider. InputForm emits
   precision marks (1.5`30.) and *^ exponents; both are stripped/translated
   into plain scientific text that every strtod-family parser accepts. *)
toNumberString[x_] := StringReplace[
    ToString[x, InputForm],
    {"`" ~~ Shortest[___] ~~ "*^" ~~ e : (("-" | DigitCharacter) ..) :> "e" <> e,
     "`" ~~ Shortest[___] ~~ EndOfString :> "",
     "*^" ~~ e : (("-" | DigitCharacter) ..) :> "e" <> e}];

encodeNumber[x_, "Float64"] := N[x];
encodeNumber[x_, precision_String] :=
    toNumberString[N[x, precisionDigits[precision]]];

precisionDigits["Float64x2"] = 34;
precisionDigits["Float64x4"] = 66;
precisionDigits["BigFloat"] = 80;   (* re-parsed at precision_bits on the Julia side *)
precisionDigits[_] = 17;

(* Result numbers come back as strings; *^ is Mathematica's own exponent. *)
fromNumberString[s_String] := ToExpression[StringReplace[s, "e" -> "*^"]];
fromNumberString[x_?NumericQ] := x;
fromNumberString[Null] := Missing["NotAvailable"];

(* ---- matrix encoding ----------------------------------------------------- *)

(* Sparse COO from either a SparseArray or a dense matrix: only structural
   nonzeros cross the boundary. Indices are 1-based on both sides. *)
cooEntries[matrix_SparseArray, precision_] := Module[{rules},
    rules = Most[ArrayRules[matrix]];
    <|"rows" -> rules[[All, 1, 1]], "cols" -> rules[[All, 1, 2]],
      "values" -> (encodeNumber[#, precision] & /@ rules[[All, 2]])|>];
cooEntries[matrix_List, precision_] :=
    cooEntries[SparseArray[matrix], precision];

encodeBlock[coefficients_List, constant_, precision_] := Module[{dim},
    dim = Length[constant];
    <|"dimension" -> dim,
      "constant" -> cooEntries[constant, precision],
      "coefficients" -> MapIndexed[
          Append[cooEntries[#1, precision], "variable" -> First[#2]] &,
          coefficients]|>];

(* ---- the bridge ---------------------------------------------------------- *)

Options[SDPXOptimize] = {
    "Precision" -> "Float64",
    "PrecisionBits" -> 256,
    "Tolerance" -> Automatic,
    "MaxIterations" -> 200,
    "TimeLimit" -> Infinity,
    "Verbosity" -> 0,
    "ReturnMatrices" -> False,
    "JuliaExecutable" -> "julia",
    "SDPXDirectory" -> Automatic,
    "KeepFiles" -> False};

(* Both public signatures route to one private implementation. They must not
   call each other: an empty list matches OptionsPattern[], so
   SDPXOptimize[c, A, C, {}, {}] would re-match the three-argument form and
   recurse forever -- observed as an $IterationLimit hang on first test. *)
SDPXOptimize[c_List, A_List, Cmats_List, opts : OptionsPattern[]] :=
    iSDPXOptimize[c, A, Cmats, {}, {}, opts];

SDPXOptimize[c_List, A_List, Cmats_List, B_, b_List, opts : OptionsPattern[]] :=
    iSDPXOptimize[c, A, Cmats, B, b, opts];

Options[iSDPXOptimize] = Options[SDPXOptimize];

iSDPXOptimize[c_List, A_List, Cmats_List, B_, b_List, OptionsPattern[]] :=
  Module[{precision, root, script, problem, settings, inFile, outFile, process,
          result, cleanup, keepFiles, tolerance},
    keepFiles = TrueQ[OptionValue["KeepFiles"]];
    precision = OptionValue["Precision"];
    root = Replace[OptionValue["SDPXDirectory"], Automatic :> $packageRoot];
    script = FileNameJoin[{root, "bin", "sdpx_solve.jl"}];
    If[!FileExistsQ[script],
        Message[SDPXOptimize::setup, script];
        Return[Failure["SDPXSetup", <|"MessageTemplate" -> "missing " <> script|>]]];

    tolerance = Replace[OptionValue["Tolerance"],
        Automatic :> If[precision === "Float64", "1e-8", "1e-20"]];
    settings = <|
        "tolerance" -> If[NumericQ[tolerance], toNumberString[tolerance],
                          tolerance],
        "maximum_iterations" -> OptionValue["MaxIterations"],
        "verbosity" -> OptionValue["Verbosity"],
        "precision_bits" -> OptionValue["PrecisionBits"],
        "return_matrices" -> TrueQ[OptionValue["ReturnMatrices"]]|>;
    If[OptionValue["TimeLimit"] =!= Infinity,
        settings["time_limit"] = N[OptionValue["TimeLimit"]]];

    problem = <|
        "sdpx_schema" -> 1,
        "precision" -> precision,
        "objective" -> (encodeNumber[#, precision] & /@ c),
        "blocks" -> MapThread[encodeBlock[#1, #2, precision] &, {A, Cmats}],
        "settings" -> settings|>;
    If[Length[b] > 0, Module[{rules},
        rules = Most[ArrayRules[SparseArray[B]]];
        problem["equalities"] = <|
            "rows" -> rules[[All, 1, 1]], "cols" -> rules[[All, 1, 2]],
            "values" -> (encodeNumber[#, precision] & /@ rules[[All, 2]]),
            "rhs" -> (encodeNumber[#, precision] & /@ b)|>]];

    inFile = FileNameJoin[{$TemporaryDirectory,
        "sdpx-" <> CreateUUID[] <> "-in.json"}];
    outFile = FileNameJoin[{$TemporaryDirectory,
        "sdpx-" <> CreateUUID[] <> "-out.json"}];
    cleanup := If[!keepFiles,
        Quiet[DeleteFile /@ Select[{inFile, outFile}, FileExistsQ]]];

    Export[inFile, problem, "JSON", "Compact" -> True];
    process = RunProcess[
        {OptionValue["JuliaExecutable"], "--startup-file=no",
         "--project=" <> FileNameJoin[{root, "bin"}], script, inFile, outFile}];

    If[!FileExistsQ[outFile],
        cleanup;
        Message[SDPXOptimize::noresult, process["StandardError"]];
        Return[Failure["SDPXBridge",
            <|"MessageTemplate" -> "no result file",
              "ExitCode" -> process["ExitCode"],
              "StandardError" -> process["StandardError"]|>]]];

    result = Import[outFile, "RawJSON"];
    cleanup;

    If[!TrueQ[result["success"]],
        Message[SDPXOptimize::solver, result["error"]];
        Return[Failure["SDPXError",
            <|"MessageTemplate" -> result["error"],
              "Status" -> Lookup[result, "status", "Error"]|>]]];

    <|"Status" -> result["status"],
      "Optimal" -> TrueQ[result["optimal"]],
      "Objective" -> fromNumberString[result["objective"]],
      "DualObjective" -> fromNumberString[result["dual_objective"]],
      "RelativeGap" -> fromNumberString[result["relative_gap"]],
      "PrimalResidual" -> fromNumberString[result["primal_residual"]],
      "DualResidual" -> fromNumberString[result["dual_residual"]],
      "Iterations" -> result["iterations"],
      "x" -> (fromNumberString /@ result["x"]),
      "y" -> (fromNumberString /@ result["y"]),
      "Certificate" -> If[KeyExistsQ[result, "certificate"],
          Association[result["certificate"]], Missing["NotRequested"]],
      "X" -> If[KeyExistsQ[result, "X"],
          MapThread[ArrayReshape[fromNumberString /@ #1, {#2, #2}] &,
              {result["X"], result["block_dimensions"]}],
          Missing["NotRequested"]],
      "Y" -> If[KeyExistsQ[result, "Y"],
          MapThread[ArrayReshape[fromNumberString /@ #1, {#2, #2}] &,
              {result["Y"], result["block_dimensions"]}],
          Missing["NotRequested"]],
      "Message" -> Lookup[result, "message", ""]|>];

End[];
EndPackage[];
