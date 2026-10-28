# Checkpoint 19750 versus 17574: verified scripted-opponent comparison

The replacement completed all **81,920 games**, **20,480 paired roots**, **1,280 calls**, and **16 stages**, with zero missing, unfinished or incomplete observations. Independent unchanged CPU audit: exit 0, PASS / CONTENT_ONLY.

## Wins by 160 hands

| Stratum | Baseline 17574 W/L/D/U | Candidate 19750 W/L/D/U | Baseline win | Candidate win | Change (percentage points), nominal 95% paired CI |
|---|---:|---:|---:|---:|---:|
| development (16 styles) | 7648/544/0/0 | 7536/656/0/0 | 93.3594% | 91.9922% | -1.3672 [-2.0752, -0.6226] |
| training_excluded_exposed_confirmation (64 styles) | 32013/755/0/0 | 31832/936/0/0 | 97.6959% | 97.1436% | -0.5524 [-0.7782, -0.3235] |

Candidate win rate is lower in both predeclared strata. These are paired comparisons against scripted/formula opponents, not direct games against the previous policy. Do not pool the exposure strata for primary inference.

## development bid and coordination metrics

BN means blind nil; the source calls its declaration counters `double_nil`. Rates below distinguish declaration opportunities, individual zero tricks, and whole-team success.

| Metric | Baseline numerator/denominator (rate) | Candidate numerator/denominator (rate) | Candidate−baseline pp [nominal 95% paired CI] |
|---|---:|---:|---:|
| positive_bid_individual_coverage | 124260/140907 (88.1858%) | 123209/140692 (87.5736%) | -0.6123 [-0.8261, -0.4015] |
| positive_bid_team_contract_success | 135640/140907 (96.2621%) | 134346/140692 (95.4894%) | -0.7726 [-0.9619, -0.5878] |
| nil_declaration_rate | 14061/154968 (9.0735%) | 14140/154832 (9.1325%) | +0.0590 [-0.0804, +0.2023] |
| nil_zero_trick_success | 11907/14061 (84.6810%) | 11947/14140 (84.4908%) | -0.1902 [-0.9357, +0.5708] |
| nil_whole_team_success | 11694/14061 (83.1662%) | 11698/14140 (82.7298%) | -0.4364 [-1.2513, +0.3769] |
| double_nil_declaration_rate | 564/6626 (8.5119%) | 244/7718 (3.1614%) | -5.3505 [-7.5936, -3.1760] |
| double_nil_zero_trick_success | 13/564 (2.3050%) | 13/244 (5.3279%) | +3.0229 [+0.1158, +6.6899] |
| double_nil_whole_team_success | 0/564 (0.0000%) | 0/244 (0.0000%) | +0.0000 [+0.0000, +0.0000] |
| bn_bn_per_eligible_hand | 278/3313 (8.3912%) | 122/3859 (3.1614%) | -5.2297 [-7.4657, -3.0626] |
| team_contract_success | 73667/77766 (94.7291%) | 73022/77538 (94.1758%) | -0.5533 [-0.8248, -0.2873] |

| Arm | Dual BN / any BN hands | Dual BN individual zeros / declarations | Dual BN both-zero / hands | Dual BN team-success / hands | BN+nil hands / both-zero / team-success | Nil+nil hands / both-zero / team-success |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 278/286 | 13/556 | 0/278 | 0/278 | 0 / 0 / 0 | 2 / 0 / 0 |
| candidate | 122/122 | 13/244 | 0/122 | 0/122 | 0 / 0 / 0 | 10 / 3 / 3 |

## training_excluded_exposed_confirmation bid and coordination metrics

BN means blind nil; the source calls its declaration counters `double_nil`. Rates below distinguish declaration opportunities, individual zero tricks, and whole-team success.

| Metric | Baseline numerator/denominator (rate) | Candidate numerator/denominator (rate) | Candidate−baseline pp [nominal 95% paired CI] |
|---|---:|---:|---:|
| positive_bid_individual_coverage | 571626/687564 (83.1379%) | 590857/722903 (81.7339%) | -1.4039 [-1.5166, -1.2938] |
| positive_bid_team_contract_success | 618672/687564 (89.9803%) | 635270/722903 (87.8776%) | -2.1027 [-2.2530, -1.9539] |
| nil_declaration_rate | 46194/733758 (6.2955%) | 49407/772310 (6.3973%) | +0.1018 [+0.0390, +0.1653] |
| nil_zero_trick_success | 39542/46194 (85.5999%) | 41927/49407 (84.8604%) | -0.7394 [-1.1627, -0.3186] |
| nil_whole_team_success | 37716/46194 (81.6470%) | 39458/49407 (79.8632%) | -1.7838 [-2.2729, -1.2986] |
| double_nil_declaration_rate | 8098/27346 (29.6131%) | 6922/34060 (20.3230%) | -9.2901 [-13.7843, -4.8383] |
| double_nil_zero_trick_success | 402/8098 (4.9642%) | 504/6922 (7.2811%) | +2.3169 [+1.5475, +3.0894] |
| double_nil_whole_team_success | 0/8098 (0.0000%) | 6/6922 (0.0867%) | +0.0867 [+0.0000, +0.1980] |
| bn_bn_per_eligible_hand | 4034/13673 (29.5034%) | 3460/17030 (20.3171%) | -9.1863 [-13.6732, -4.7257] |
| team_contract_success | 328194/370928 (88.4792%) | 337367/389616 (86.5896%) | -1.8895 [-2.1803, -1.6054] |

| Arm | Dual BN / any BN hands | Dual BN individual zeros / declarations | Dual BN both-zero / hands | Dual BN team-success / hands | BN+nil hands / both-zero / team-success | Nil+nil hands / both-zero / team-success |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4034/4064 | 402/8068 | 0/4034 | 0/4034 | 1 / 0 / 0 | 10 / 1 / 1 |
| candidate | 3460/3462 | 504/6920 | 3/3460 | 3/3460 | 0 / 0 / 0 | 31 / 6 / 6 |

The JSON retains all eligible/conditional partner-state exposures, raw counts, paired intervals and bid histograms. Two declarations occur per dual-BN hand. Fewer declarations alone do not establish improvement when eligibility changes. No forced-bid rule was introduced.

## Hardware and timing evidence

Recorded two distinct Tesla T4 UUID/PCI bindings, balanced complete-pair shards and unchanged native parameter replicas. The content auditor verifies recorded provenance; independent provider settings, terminal status and deadline observations remain separate parent evidence. Do not claim 100% GPU utilization or substitute device visibility for execution.

## Limitations

- Both fixed checkpoints played the same scripted/formula opponent roster; this is not head-to-head or a self-play win rate.
- The 64 training-excluded styles were previously evaluated, not a fresh unseen holdout.
- Baseline-only comparison cannot distinguish the hand-clock change from additional training.
- Deterministic argmax results are not stochastic training policy rates.
- Intervals condition on fixed checkpoints, styles and 160-hand horizon; they exclude training-seed uncertainty.
- Zero-event bootstrap intervals do not prove zero population probability; zero-denominator rates remain null.
- Private evidence and model binaries are not included in this results directory.
