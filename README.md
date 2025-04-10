1. MCTrack 리퍼지토리 설치 & 가상환경 생성
2. conda activate mctrack(가상환경이름)
3. roscore
4. tracker_ws catkin_make 해주기
5. ~/tracker_ws/src/mctrack_tracker/scripts$ python convert_nuscenes_to_results.py 로 valsplit gt json파일생성 
6. ~/tracker_ws$ roslaunch mctrack_tracker run_eval.launch 실행
7. ~/tracker_ws/src/mctrack_tracker/scripts$ python filter_results.py   --input ~/nuscenes_tracking_results.json   --output ~/SOTA/MCTrack/results/nuscenes/mctrack_custom/results.json 트래킹 결과 MCTrack 평가기준에 맞춰 필터링
8. ~/SOTA/MCTrack$ python evaluation/static_evaluation/nuscenes/eval.py results/nuscenes/mctrack_custom/results.json 필터링된 결과를 MCTrack 내부 평가 툴킷으로 성능평가

   
<현재 tracker 성능(Updated 25.04.10)>
### Final results (Before Hungarian Matching)

| Class     | AMOTA | AMOTP | RECALL | MOTAR | GT        | MOTA  | MOTP  | MT   | ML  | FAF  | TP     | FP     | FN     | IDS | FRAG | TID  | LGD  |
|-----------|-------|-------|--------|--------|-----------|-------|-------|------|-----|------|--------|--------|--------|-----|------|------|------|
| bicycle   | 0.643 | 0.354 | 0.849  | 0.779  | 19930.657 | 0.004 | 97    | 11   | 25.4 | 1681 | 371300 | 12     | 9      | 0.76| 0.81 |
| bus       | 0.645 | 0.502 | 0.808  | 0.860  | 21120.679 | 0.002 | 69    | 18   | 15.2 | 1666 | 233406 | 40     | 21     | 0.85| 1.55 |
| car       | 0.573 | 0.506 | 0.805  | 0.763  | 58317     | 0.600 | 0.008 | 2304 | 544  | 188.5| 45870  | 10853  | 11373  |1074 | 477  | 0.77 | 1.04 |
| motorcy   | 0.492 | 0.566 | 0.785  | 0.679  | 19770.510 | 0.022 | 78    | 20   | 35.0 | 1486 | 477426 | 65     | 29     | 0.77| 1.02 |
| pedestr   | 0.689 | 0.402 | 0.895  | 0.810  | 25423     | 0.711 | 0.120 | 1272 | 75   | 97.4 | 22309  | 4237   | 2662   | 452 | 265  | 0.60 | 0.66 |
| trailer   | 0.746 | 0.254 | 0.923  | 0.852  | 24250.772 | 0.005 | 107   | 12   | 32.9 | 2198 | 325186 | 41     | 21     | 0.45| 0.56 |
| truck     | 0.734 | 0.352 | 0.867  | 0.890  | 96500.764 | 0.003 | 397   | 68   | 25.1 | 8281 | 9131282| 87     | 57     | 0.56| 0.74 |

**Aggregated results:**

- **AMOTA**: 0.646
- **AMOTP**: 0.419
- **RECALL**: 0.847
- **MOTAR**: 0.805
- **GT**: 14556
- **MOTA**: 0.670
- **MOTP**: 0.023
- **MT**: 4324
- **ML**: 748
- **FAF**: 59.9
- **TP**: 83491
- **FP**: 17409
- **FN**: 16635
- **IDS**: 1771
- **FRAG**: 879
- **TID**: 0.68
- **LGD**: 0.91
- **Eval time**: 96.1s



<참고>
mctrack_tracker_node_base.py >  칼만필터만 적용된 베이스 트래커, 성능은 이하와 같음
### Final results

| Class     | AMOTA | AMOTP | RECALL | MOTAR |   GT     | MOTA | MOTP | MT   | ML |  FAF  |   TP   |    FP     |  FN  | IDS | FRAG |  TID |  LGD |
|-----------|-------|-------|--------|-------|----------|------|------|------|----|-------|--------|-----------|------|-----|------|------|------|
| bicycle   | 0.647 | 0.104 | 0.998  | 0.681 | 19930.659| 0.004|      | 156  | 0  |  41.3 |  1927  |   614     | 462  |  3  |      | 0.00 | 0.01 |
| bus       | 0.225 | 0.353 | 0.983  | 0.273 | 21120.233| 0.004|      | 106  | 0  |  84.4 |  1803  | 131135    |      |274  |  4   | 0.00 | 0.11 |
| car       | 0.170 | 0.308 | 0.994  | 0.200 |583170.174| 0.009|      | 3670 | 0  | 702.8 | 50712  |40554338   |      |7267 | 123  | 0.00 | 0.04 |
| motorcycle| 0.041 | 0.365 | 0.998  | 0.049 | 19770.041| 0.019|      | 132  | 0  | 111.4 |  1659  |15774314   |      | 2   |      | 0.00 | 0.01 |
| pedestrian| 0.764 | 0.212 | 0.991  | 0.804 |254230.780| 0.118|      | 1684 | 0  | 110.8 | 24662  | 4830240   |      |521  |  98  | 0.00 | 0.05 |
| trailer   | 0.703 | 0.105 | 1.000  | 0.740 | 24250.710| 0.006|      | 133  | 0  |  60.6 |  2327  |   606     |  98  | 0   |      | 0.00 | 0.00 |
| truck     | 0.484 | 0.203 | 0.999  | 0.538 | 96500.494| 0.003|      | 541  | 0  | 110.8 |  8852  |  408811   |      | 787 |  5   | 0.00 | 0.01 |


| Metric | Value  |
|--------|--------|
| AMOTA  | 0.433  |
| AMOTP  | 0.236  |
| RECALL | 0.995  |
| MOTAR  | 0.469  |
| GT     | 14556  |
| MOTA   | 0.442  |
| MOTP   | 0.023  |
| MT     | 6422   |
| ML     | 0      |
| FAF    | 174.6  |
| TP     | 91942  |
| FP     | 53580  |
| FN     | 632    |
| IDS    | 9323   |
| FRAG   | 235    |
| TID    | 0.00   |
| LGD    | 0.03   |
| Eval time | 115.5s |
 



mctrack_tracker_node.py 수정하면서 성능 높이면 될 듯
