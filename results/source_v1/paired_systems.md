# 来源逐系统正反结果

每行是一个独立系统，优化seed已在系统内平均。差值为百分点；含全部负例，不按本表筛选目标样本。

## temporal_vs_node

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | +9.006 | -13.889 | -7.342 |
| confirmation_001 | 12 | tanh | +5.251 | -11.111 | +0.294 |
| confirmation_002 | 16 | sin | +1.034 | +38.889 | -0.336 |
| confirmation_003 | 20 | linear | +5.982 | -16.667 | +5.385 |
| confirmation_004 | 8 | tanh | -9.137 | +0.000 | +0.109 |
| confirmation_005 | 12 | sin | +5.217 | +22.222 | -1.957 |
| confirmation_006 | 16 | linear | -12.627 | -8.333 | +0.403 |
| confirmation_007 | 20 | tanh | +5.563 | +16.667 | +1.008 |
| confirmation_008 | 8 | sin | +10.563 | +25.000 | -0.286 |
| confirmation_009 | 12 | linear | +0.839 | +27.778 | +0.395 |
| confirmation_010 | 16 | tanh | -12.938 | -8.333 | -3.049 |
| confirmation_011 | 20 | sin | +0.808 | +0.000 | +0.899 |
| confirmation_012 | 8 | linear | +0.957 | -19.444 | +0.034 |
| confirmation_013 | 12 | tanh | +0.077 | +22.222 | +0.445 |
| confirmation_014 | 16 | sin | +4.774 | +16.667 | -2.310 |
| confirmation_015 | 20 | linear | +4.184 | -33.333 | -0.403 |
| confirmation_016 | 8 | tanh | -16.584 | -11.111 | +0.227 |
| confirmation_017 | 12 | sin | +2.685 | +8.333 | -11.349 |
| confirmation_018 | 16 | linear | -3.562 | +13.889 | -1.226 |
| confirmation_019 | 20 | tanh | +4.764 | +8.333 | -1.562 |
| confirmation_020 | 8 | sin | +6.469 | +8.333 | -0.344 |
| confirmation_021 | 12 | linear | -2.480 | +0.000 | +0.076 |
| confirmation_022 | 16 | tanh | -2.006 | -5.556 | +0.487 |
| confirmation_023 | 20 | sin | +5.633 | +16.667 | -0.017 |
| confirmation_024 | 8 | linear | -1.036 | +16.667 | +0.361 |
| confirmation_025 | 12 | tanh | +5.176 | +25.000 | +0.277 |
| confirmation_026 | 16 | sin | +3.815 | +55.556 | -3.125 |
| confirmation_027 | 20 | linear | +17.005 | -11.111 | -0.706 |
| confirmation_028 | 8 | tanh | -8.487 | +8.333 | +0.143 |
| confirmation_029 | 12 | sin | +1.849 | +8.333 | +0.932 |
| confirmation_030 | 16 | linear | +17.695 | +0.000 | +0.134 |
| confirmation_031 | 20 | tanh | +7.282 | +8.333 | +0.244 |

## temporal_vs_full

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | -0.024 | -22.222 | +9.241 |
| confirmation_001 | 12 | tanh | -4.904 | +5.556 | -0.487 |
| confirmation_002 | 16 | sin | +0.446 | +13.889 | -0.437 |
| confirmation_003 | 20 | linear | +23.589 | -25.000 | +7.502 |
| confirmation_004 | 8 | tanh | +26.930 | +0.000 | -0.848 |
| confirmation_005 | 12 | sin | +0.190 | +22.222 | -0.798 |
| confirmation_006 | 16 | linear | -0.057 | +0.000 | -0.126 |
| confirmation_007 | 20 | tanh | +1.626 | +16.667 | +0.403 |
| confirmation_008 | 8 | sin | +6.428 | +33.333 | -0.890 |
| confirmation_009 | 12 | linear | +15.218 | +52.778 | -0.840 |
| confirmation_010 | 16 | tanh | +22.389 | +0.000 | -0.756 |
| confirmation_011 | 20 | sin | -2.663 | +8.333 | -0.286 |
| confirmation_012 | 8 | linear | -0.969 | +22.222 | -0.596 |
| confirmation_013 | 12 | tanh | +5.248 | +5.556 | -0.487 |
| confirmation_014 | 16 | sin | +5.767 | +33.333 | -0.370 |
| confirmation_015 | 20 | linear | +20.868 | -33.333 | +0.101 |
| confirmation_016 | 8 | tanh | +25.331 | -19.444 | -0.680 |
| confirmation_017 | 12 | sin | +1.131 | +8.333 | -0.134 |
| confirmation_018 | 16 | linear | +11.380 | +30.556 | +0.034 |
| confirmation_019 | 20 | tanh | +8.821 | +8.333 | +0.025 |
| confirmation_020 | 8 | sin | -3.946 | -8.333 | -0.748 |
| confirmation_021 | 12 | linear | +11.765 | +16.667 | -0.554 |
| confirmation_022 | 16 | tanh | +24.437 | -5.556 | -0.571 |
| confirmation_023 | 20 | sin | -2.195 | +16.667 | -0.370 |
| confirmation_024 | 8 | linear | +8.979 | +25.000 | +0.260 |
| confirmation_025 | 12 | tanh | +10.948 | +41.667 | -0.454 |
| confirmation_026 | 16 | sin | +0.728 | +55.556 | -0.277 |
| confirmation_027 | 20 | linear | +15.701 | -11.111 | -0.252 |
| confirmation_028 | 8 | tanh | +22.111 | +0.000 | -0.487 |
| confirmation_029 | 12 | sin | -5.891 | +8.333 | -0.126 |
| confirmation_030 | 16 | linear | +12.991 | +0.000 | -0.672 |
| confirmation_031 | 20 | tanh | +7.681 | -8.333 | -0.638 |

## temporal_graph

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | +6.961 | +0.000 | +8.434 |
| confirmation_001 | 12 | tanh | -6.905 | -38.889 | +0.134 |
| confirmation_002 | 16 | sin | -11.582 | +22.222 | +1.193 |
| confirmation_003 | 20 | linear | +0.101 | +8.333 | +2.781 |
| confirmation_004 | 8 | tanh | +12.777 | +0.000 | +0.008 |
| confirmation_005 | 12 | sin | -11.641 | -27.778 | +0.067 |
| confirmation_006 | 16 | linear | +7.109 | +11.111 | -0.504 |
| confirmation_007 | 20 | tanh | +2.219 | +16.667 | +0.655 |
| confirmation_008 | 8 | sin | -3.114 | -8.333 | -0.218 |
| confirmation_009 | 12 | linear | +3.157 | -5.556 | -0.370 |
| confirmation_010 | 16 | tanh | +18.464 | +0.000 | +0.252 |
| confirmation_011 | 20 | sin | -3.453 | +8.333 | +0.731 |
| confirmation_012 | 8 | linear | +0.418 | +38.889 | -0.067 |
| confirmation_013 | 12 | tanh | +2.339 | +2.778 | -0.067 |
| confirmation_014 | 16 | sin | +3.554 | +30.556 | +0.748 |
| confirmation_015 | 20 | linear | +17.365 | +0.000 | +0.193 |
| confirmation_016 | 8 | tanh | +5.649 | +8.333 | -0.101 |
| confirmation_017 | 12 | sin | +4.844 | +16.667 | +1.176 |
| confirmation_018 | 16 | linear | +2.517 | -2.778 | +0.479 |
| confirmation_019 | 20 | tanh | +17.664 | +33.333 | +0.521 |
| confirmation_020 | 8 | sin | -8.696 | -33.333 | +0.454 |
| confirmation_021 | 12 | linear | +7.897 | +61.111 | -0.311 |
| confirmation_022 | 16 | tanh | +32.608 | +19.444 | +0.353 |
| confirmation_023 | 20 | sin | -8.018 | +16.667 | +0.630 |
| confirmation_024 | 8 | linear | +26.572 | +27.778 | -2.764 |
| confirmation_025 | 12 | tanh | +4.683 | +38.889 | -0.546 |
| confirmation_026 | 16 | sin | -3.345 | +72.222 | +1.369 |
| confirmation_027 | 20 | linear | +1.598 | +2.778 | +0.034 |
| confirmation_028 | 8 | tanh | -1.697 | +5.556 | -0.067 |
| confirmation_029 | 12 | sin | -2.559 | +8.333 | +1.327 |
| confirmation_030 | 16 | linear | +7.645 | +2.778 | -0.470 |
| confirmation_031 | 20 | tanh | -1.486 | -2.778 | +0.126 |

## edge_vs_node_all

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | +1.450 | +25.000 | -8.090 |
| confirmation_001 | 12 | tanh | -2.217 | -25.000 | +1.966 |
| confirmation_002 | 16 | sin | -5.832 | -41.667 | -1.613 |
| confirmation_003 | 20 | linear | +4.841 | +8.333 | -1.991 |
| confirmation_004 | 8 | tanh | -18.193 | -41.667 | +0.630 |
| confirmation_005 | 12 | sin | -1.439 | -16.667 | -2.747 |
| confirmation_006 | 16 | linear | -9.211 | -25.000 | +0.983 |
| confirmation_007 | 20 | tanh | -9.521 | +0.000 | +0.000 |
| confirmation_008 | 8 | sin | +5.701 | +16.667 | -0.781 |
| confirmation_009 | 12 | linear | -6.532 | +0.000 | +0.000 |
| confirmation_010 | 16 | tanh | -17.377 | -25.000 | +6.048 |
| confirmation_011 | 20 | sin | -0.049 | -8.333 | -0.504 |
| confirmation_012 | 8 | linear | +2.582 | +8.333 | +0.000 |
| confirmation_013 | 12 | tanh | +2.980 | +8.333 | +0.000 |
| confirmation_014 | 16 | sin | +3.384 | -8.333 | -1.285 |
| confirmation_015 | 20 | linear | -12.726 | +0.000 | +3.427 |
| confirmation_016 | 8 | tanh | -19.679 | -50.000 | +0.000 |
| confirmation_017 | 12 | sin | -1.384 | -8.333 | -9.299 |
| confirmation_018 | 16 | linear | -0.979 | -41.667 | +0.202 |
| confirmation_019 | 20 | tanh | -10.730 | -50.000 | +0.706 |
| confirmation_020 | 8 | sin | -2.702 | -16.667 | -1.159 |
| confirmation_021 | 12 | linear | -5.700 | -41.667 | +0.000 |
| confirmation_022 | 16 | tanh | +2.035 | -16.667 | +0.000 |
| confirmation_023 | 20 | sin | +3.334 | -16.667 | +2.344 |
| confirmation_024 | 8 | linear | -13.739 | -66.667 | +1.336 |
| confirmation_025 | 12 | tanh | +0.138 | -33.333 | +2.041 |
| confirmation_026 | 16 | sin | +5.869 | +0.000 | -1.638 |
| confirmation_027 | 20 | linear | +13.305 | -66.667 | -0.882 |
| confirmation_028 | 8 | tanh | -4.470 | -58.333 | -0.025 |
| confirmation_029 | 12 | sin | -8.551 | +8.333 | -0.630 |
| confirmation_030 | 16 | linear | +2.733 | +8.333 | +0.000 |
| confirmation_031 | 20 | tanh | -1.165 | -33.333 | -0.025 |

## protected_vs_raw

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | -0.148 | — | +0.193 |
| confirmation_001 | 12 | tanh | -1.553 | — | +0.101 |
| confirmation_002 | 16 | sin | +2.277 | — | +0.076 |
| confirmation_003 | 20 | linear | +4.491 | — | +0.034 |
| confirmation_004 | 8 | tanh | +4.869 | — | -0.034 |
| confirmation_005 | 12 | sin | -3.041 | — | -0.042 |
| confirmation_006 | 16 | linear | +0.044 | — | -0.059 |
| confirmation_007 | 20 | tanh | +1.156 | — | -0.050 |
| confirmation_008 | 8 | sin | +1.636 | — | -0.101 |
| confirmation_009 | 12 | linear | +0.586 | — | +0.050 |
| confirmation_010 | 16 | tanh | +4.620 | — | +0.059 |
| confirmation_011 | 20 | sin | -1.242 | — | +0.000 |
| confirmation_012 | 8 | linear | -0.266 | — | +0.067 |
| confirmation_013 | 12 | tanh | +0.600 | — | -0.076 |
| confirmation_014 | 16 | sin | +0.843 | — | +0.017 |
| confirmation_015 | 20 | linear | +4.344 | — | +0.000 |
| confirmation_016 | 8 | tanh | +3.406 | — | +0.126 |
| confirmation_017 | 12 | sin | -0.612 | — | -0.025 |
| confirmation_018 | 16 | linear | +1.721 | — | +0.042 |
| confirmation_019 | 20 | tanh | +4.247 | — | +0.319 |
| confirmation_020 | 8 | sin | -1.757 | — | +0.092 |
| confirmation_021 | 12 | linear | +3.668 | — | -0.025 |
| confirmation_022 | 16 | tanh | +2.619 | — | +0.092 |
| confirmation_023 | 20 | sin | -0.889 | — | +0.050 |
| confirmation_024 | 8 | linear | +5.153 | — | -0.017 |
| confirmation_025 | 12 | tanh | +4.112 | — | -0.160 |
| confirmation_026 | 16 | sin | -0.208 | — | +0.000 |
| confirmation_027 | 20 | linear | +7.258 | — | +0.059 |
| confirmation_028 | 8 | tanh | +0.347 | — | -0.025 |
| confirmation_029 | 12 | sin | -1.378 | — | +0.008 |
| confirmation_030 | 16 | linear | +2.335 | — | -0.042 |
| confirmation_031 | 20 | tanh | +3.058 | — | +0.025 |

## protected_vs_wrong

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | -0.116 | — | +0.076 |
| confirmation_001 | 12 | tanh | -0.692 | — | +0.126 |
| confirmation_002 | 16 | sin | +1.268 | — | +0.076 |
| confirmation_003 | 20 | linear | +3.044 | — | -0.017 |
| confirmation_004 | 8 | tanh | +3.586 | — | -0.008 |
| confirmation_005 | 12 | sin | -1.853 | — | -0.126 |
| confirmation_006 | 16 | linear | -0.091 | — | +0.042 |
| confirmation_007 | 20 | tanh | +0.607 | — | +0.067 |
| confirmation_008 | 8 | sin | +0.791 | — | +0.101 |
| confirmation_009 | 12 | linear | +0.741 | — | +0.017 |
| confirmation_010 | 16 | tanh | +2.956 | — | +0.034 |
| confirmation_011 | 20 | sin | -0.624 | — | +0.000 |
| confirmation_012 | 8 | linear | -0.155 | — | -0.008 |
| confirmation_013 | 12 | tanh | +0.502 | — | -0.050 |
| confirmation_014 | 16 | sin | +0.394 | — | +0.017 |
| confirmation_015 | 20 | linear | +2.744 | — | +0.000 |
| confirmation_016 | 8 | tanh | +2.519 | — | +0.042 |
| confirmation_017 | 12 | sin | +0.112 | — | +0.017 |
| confirmation_018 | 16 | linear | +0.998 | — | +0.042 |
| confirmation_019 | 20 | tanh | +0.201 | — | +0.344 |
| confirmation_020 | 8 | sin | -1.900 | — | +0.059 |
| confirmation_021 | 12 | linear | +2.752 | — | +0.025 |
| confirmation_022 | 16 | tanh | +2.552 | — | +0.101 |
| confirmation_023 | 20 | sin | -0.710 | — | +0.050 |
| confirmation_024 | 8 | linear | +4.236 | — | -0.008 |
| confirmation_025 | 12 | tanh | +3.885 | — | -0.160 |
| confirmation_026 | 16 | sin | -0.421 | — | +0.025 |
| confirmation_027 | 20 | linear | +2.524 | — | +0.067 |
| confirmation_028 | 8 | tanh | -0.471 | — | -0.059 |
| confirmation_029 | 12 | sin | -0.094 | — | +0.008 |
| confirmation_030 | 16 | linear | +1.411 | — | -0.076 |
| confirmation_031 | 20 | tanh | +2.204 | — | +0.025 |

## typed_vs_temporal

| 系统 | C | 动力学 | AP差 | 条件Top1差 | 正常FPR差 |
|---|---:|---|---:|---:|---:|
| confirmation_000 | 8 | linear | +0.000 | +27.778 | +0.000 |
| confirmation_001 | 12 | tanh | +0.000 | -16.667 | +0.000 |
| confirmation_002 | 16 | sin | +0.000 | -2.778 | +0.000 |
| confirmation_003 | 20 | linear | +0.000 | +0.000 | +0.000 |
| confirmation_004 | 8 | tanh | +0.000 | +2.778 | +0.000 |
| confirmation_005 | 12 | sin | +0.000 | -5.556 | +0.000 |
| confirmation_006 | 16 | linear | +0.000 | -13.889 | +0.000 |
| confirmation_007 | 20 | tanh | +0.000 | -5.556 | +0.000 |
| confirmation_008 | 8 | sin | +0.000 | -16.667 | +0.000 |
| confirmation_009 | 12 | linear | +0.000 | -8.333 | +0.000 |
| confirmation_010 | 16 | tanh | +0.000 | +0.000 | +0.000 |
| confirmation_011 | 20 | sin | +0.000 | -2.778 | +0.000 |
| confirmation_012 | 8 | linear | +0.000 | -19.444 | +0.000 |
| confirmation_013 | 12 | tanh | +0.000 | +5.556 | +0.000 |
| confirmation_014 | 16 | sin | +0.000 | -16.667 | +0.000 |
| confirmation_015 | 20 | linear | +0.000 | +8.333 | +0.000 |
| confirmation_016 | 8 | tanh | +0.000 | +2.778 | +0.000 |
| confirmation_017 | 12 | sin | +0.000 | +0.000 | +0.000 |
| confirmation_018 | 16 | linear | +0.000 | -8.333 | +0.000 |
| confirmation_019 | 20 | tanh | +0.000 | -2.778 | +0.000 |
| confirmation_020 | 8 | sin | +0.000 | -11.111 | +0.000 |
| confirmation_021 | 12 | linear | +0.000 | +0.000 | +0.000 |
| confirmation_022 | 16 | tanh | +0.000 | +0.000 | +0.000 |
| confirmation_023 | 20 | sin | +0.000 | -13.889 | +0.000 |
| confirmation_024 | 8 | linear | +0.000 | -27.778 | +0.000 |
| confirmation_025 | 12 | tanh | +0.000 | -19.444 | +0.000 |
| confirmation_026 | 16 | sin | +0.000 | -11.111 | +0.000 |
| confirmation_027 | 20 | linear | +0.000 | +2.778 | +0.000 |
| confirmation_028 | 8 | tanh | +0.000 | -16.667 | +0.000 |
| confirmation_029 | 12 | sin | +0.000 | -2.778 | +0.000 |
| confirmation_030 | 16 | linear | +0.000 | +2.778 | +0.000 |
| confirmation_031 | 20 | tanh | +0.000 | +0.000 | +0.000 |
