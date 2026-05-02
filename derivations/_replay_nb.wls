#!/usr/bin/env wolframscript
(* Replay both Input cells of posterior_rederivation.nb directly,
   to confirm the fixed notebook now produces the claimed results. *)

(* ===== Cell 3 (assumptions) ===== *)
$Assumptions = {\[Alpha]t > 0, \[Sigma]t > 0, \[Alpha]s > 0, \[Sigma]s > 0,
                vk > 0, t > 0, t < 1, s > 0, s < 1};
Print["Symbol count: ",
  Length[{\[Alpha]t, \[Sigma]t, \[Alpha]s, \[Sigma]s, xt, xs, \[Mu]k, vk}]];

(* ===== Cell 4 (main derivation, fixed v -> \[Nu]) ===== *)
\[Zeta] = (\[Alpha]t/\[Sigma]t)^2 - (\[Alpha]s/\[Sigma]s)^2;
\[Nu]   = \[Alpha]t*xt/\[Sigma]t^2 - \[Alpha]s*xs/\[Sigma]s^2;
dk      = vk*\[Zeta] + 1;

\[Mu]Post           = (vk*\[Nu] + \[Mu]k)/dk;
\[CapitalDelta]LogW = \[Mu]k*(\[Nu] - (1/2)*\[Zeta]*\[Mu]k)/dk;

ClearAll[x0];
logQ = -(x0 - \[Mu]k)^2/(2 vk) - (\[Zeta]/2)*x0^2 + \[Nu]*x0;
logQ = Expand[logQ];

A      = Coefficient[logQ, x0, 2];
B      = Coefficient[logQ, x0, 1];
Const0 = Coefficient[logQ, x0, 0];

logZFull    = FullSimplify[-B^2/(4 A) + Const0];
missingTerm = FullSimplify[logZFull - \[CapitalDelta]LogW];

Print["Full log-normalizer:           ", logZFull];
Print["Truncated DeltaLogW:           ", FullSimplify[\[CapitalDelta]LogW]];
Print["Missing term:                  ", missingTerm];
Print["Matches vk*Nu^2/(2*dk)? (0=yes): ",
      FullSimplify[missingTerm - vk*\[Nu]^2/(2*dk)]];
Print["Free of mu_k (k-independent)?: ", FreeQ[missingTerm, \[Mu]k]];

Print["zeta  = ", FullSimplify[\[Zeta]]];
Print["nu    = ", FullSimplify[\[Nu]]];
Print["dk    = ", FullSimplify[dk]];
Print["muBar = ", FullSimplify[\[Mu]Post]];
Print["dw    = ", FullSimplify[\[CapitalDelta]LogW]];

linSubs    = {\[Alpha]t -> 1 - \[Sigma]t, \[Alpha]s -> 1 - \[Sigma]s};
\[Zeta]Lin = FullSimplify[\[Zeta] /. linSubs];
\[Nu]Lin   = FullSimplify[\[Nu] /. linSubs];
\[Zeta]Code = ((1 - \[Sigma]t)/\[Sigma]t)^2 - ((1 - \[Sigma]s)/\[Sigma]s)^2;
\[Nu]Code   = ((1 - \[Sigma]t)/\[Sigma]t)*xt/\[Sigma]t -
              ((1 - \[Sigma]s)/\[Sigma]s)*xs/\[Sigma]s;
Print["zeta diff (must be 0): ", FullSimplify[\[Zeta]Lin - \[Zeta]Code]];
Print["nu   diff (must be 0): ", FullSimplify[\[Nu]Lin - \[Nu]Code]];

(* ===== Cell 5 (VP/trig) ===== *)
vpSubs = {\[Alpha]t -> Cos[Pi*t/2], \[Sigma]t -> Sin[Pi*t/2],
          \[Alpha]s -> Cos[Pi*s/2], \[Sigma]s -> Sin[Pi*s/2]};
\[Zeta]VP = TrigReduce[FullSimplify[\[Zeta] /. vpSubs]];
\[Nu]VP   = TrigReduce[FullSimplify[\[Nu] /. vpSubs]];
Print["zeta (VP): ", \[Zeta]VP];
Print["nu   (VP): ", \[Nu]VP];
Print["cos^2+sin^2-1 (must be 0): ",
      FullSimplify[Cos[Pi*t/2]^2 + Sin[Pi*t/2]^2 - 1]];

\[Mu]VP = \[Mu]Post /. vpSubs /. s -> 1/2;
limT0 = FullSimplify[Limit[\[Mu]VP, t -> 0, Direction -> "FromAbove"]];
limT1 = FullSimplify[Limit[\[Mu]VP, t -> 1, Direction -> "FromBelow"]];
Print["lim t->0+ (s=1/2) [expect xt]: ", limT0];
Print["lim t->1- (s=1/2) [degenerate vk=1]: ", limT1];

Print["--- TeX export ---"];
Print["zeta : ", TeXForm[\[Zeta]]];
Print["nu   : ", TeXForm[\[Nu]]];
Print["dk   : ", TeXForm[dk]];
Print["muBar: ", TeXForm[\[Mu]Post]];
Print["dw   : ", TeXForm[\[CapitalDelta]LogW]];
