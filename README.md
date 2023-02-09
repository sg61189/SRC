# SRC

A few top algorithms to understand (top 6/7 core algorithms are interesting for us):
-----------------------------------------------------------------------------------------

Min Area retiming based on maxflow-mincut algorithm (RET in Expert System paper)
https://people.eecs.berkeley.edu/~alanmi/publications/2007/fmcad07_min.pdf
 -- we have some more improvements on the retiming, but the above paper is the first in introducing the concept.
 -- We can look for retiming in this webpage for papers/improvements on retiming: https://people.eecs.berkeley.edu/~alanmi/publications/

Fast Reparameterization (CUT algorithm in Expert System)paper http://een.se/niklas/reparam.pdf
-- More improvements in input elimination: http://theory.stanford.edu/~barrett/fmcad/papers/FMCAD2019_paper_43.pdf

Redundancy identification and removal https://www.cs.utexas.edu/users/hunt/FMCAD/FMCAD11/papers/46.pdf
"Input Elimination Transformations for Scalable Verification and Trace Reconstruction." R. K. Gajavelly, J. Baumgartner. A. Ivrii, R. L. Kanzelman, and S. Ghosh. FMCAD 2019.

"Invariant Strengthened Elimination of Dependent State Elements." M. L. Case, A. Mishchenko, R. Brayton, J. Baumgartner and H. Mony, Formal Methods in Computer-Aided Design, 2008.

"Merging Nodes Under Sequential Observability." M. L. Case, V. N. Kravets, A. Mishchenko and R. Brayton, Design Automation Conference, 2008.

"Robust Boolean Reasoning for Equivalence Checking and Functional Property Verification." A. Kuehlmann, V. Paruthi, F. Krohm and M.K. Ganai, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, Volume 21, pp. 1377-1394, December 2002.

"Equivalence Checking using Cuts and Heaps." A. Kuehlmann and F. Krohm, Design Automation Conference, 1997.

"Equivalence Checking combining a Structural SAT-solver, BDDs and Simulation." V. Paruthi and A. Kuehlmann, International Conference on Computer Design, 2000.

"Circuit Based Boolean Reasoning." A. Kuehlmann, M.K. Ganai, and V. Paruthi, Design Automation Conference, 2001.

"On-the-Fly Compression of Logical Circuits." M.K. Ganai and A. Kuehlmann, International Workshop on Logic & Synthesis, 2000.

"The Use of Random Simulation in Formal Verification." F. Krohm, A. Kuehlmann, and A. Mets, International Conference on Computer Design, 1996.

"Factor Cuts." S. Chatterjee, A. Mishchenko, and R. Brayton. International Conference on Computer-Aided Design, 2006.

"DAG-Aware AIG Rewriting: A Fresh Look at Combinational Logic Synthesis." A. Mishchenko, S. Chatterjee, and R. Brayton. Design Automation Conference, 2006.

"Functional Dependency for Verification Reduction." J.-H. R. Jiang and R. Brayton. Conference on Computer-Aided Verification, 2002.

Equivalence identification and removal: 
"Speculative-Reduction based Scalable Redundancy Identification" H. Mony, J. Baumgartner, A. Mishchenko and R. K. Brayton, Design Automation and Test in Europe, 2009

Some other Algos -- Abstraction-refinement  techniques:
------------------------------------
"Automatic Abstraction without Counterexamples," K. L. McMillan and N. Amla, Tools and Algorithms for the Construction and Analysis of Systems, 2003.

"A single-instance incremental SAT formulation of proof- and counterexample-based abstraction," N. Een, A. Mishchenko, and N. Amla, FMCAD 2010.

"GLA: gate-level abstraction revisited," A. Mishchenko, N. Een, R. K. Brayton, J. Baumgartner, H. Mony, and P. K. Nalla, DATE 2013. 

Property Directed Reachability (IC3):  "Efficient Implementation of Property Directed Reachability," N. Een, A. Mishchenko, R. Brayton, FMCAD, 2011

Semiformal verification for bug hunting: "The Art of Semi-Formal Bug Hunting." P. Nalla, R. Gajavelly, J. Baumgartner, H. Mony, R. Kanzelman, A. Ivrii; ICCAD 2016.

"An effective guidance strategy for abstraction guided simulation." F. M. de Paula, A J. Hu; DAC 2007.

Some useful links:
--------------------
Link for Latest ABC: https://github.com/berkeley-abc/abc
Link for Super_prove: https://github.com/berkeley-abc/super_prove
Command to enable each of the above alogithms in ABC:  https://people.eecs.berkeley.edu/~alanmi/abc/
For hwmcc AIG benchmarks http://fmv.jku.at/hwmcc20/index.html#history

Our Working Steps:
----------------------
1. Need to understand how super_prove uses different algorithms in parallel
2. Our algorithm selection will depend upon the current design, the expected result, and property modeling/design properties. 
   At the high level we can categorize the problem as one of the following:
    (1) Falsification - find a failing counter example 
    (2) Proving the property 
    (3) Equivalence Checking
    (4) Simplification
    (5) Bug Indentification
So, our chosen set of algorithms sometimes will be very specific to the result/goal, e.g, early stage designs may be prone to have more properties falsified for bug indentification.
3. In order to map to a multi-armed bandit porblem, try to set arms based in these categories and then search for appropriate algos from each category. 
