## version of some important Python modules
1. torch: 1.13.1; numpy: 1.23.5.
2. POT: 0.8.2 (pip installation: pip install POT==0.8.2; post installation check: import ot).
3. pip install -U scimitar-learn.

## code
The folder 'pyCode' contains Python function files to implement the proposed method and other methods used for comparison.

## simulation
The file 'main_diff_signals.py' contains the code to produce the results in Figure 1.
The file 'main_diff_dimensions.py' contains the code to produce the results in Table 1.

## real data
(1) code
The file 'GBM_data_pre.R' contains the code for downloading and preprocessing the data.
The file 'GBM_methylation.py' contains the Python code to implement the proposed test in real data.
(2) files in the folder 'GBM'
The files 'GBM_prognostic_low.txt' and 'GBM_prognositic_high.txt' are the datasets used in the real data section.
The file 'prognostic_glioma.json' contains the prognostic genes related to the brain cancer. 
The file 'c5.go.bp.v2023.1.Hs.json' contains all the biological process GO terms. 
The files 'idx_genes_...' contain the indices of genes related to the specific GO term.

## visualization
The file 'plot.R' includes the code to generate the plots in simulation and real data analysis.
