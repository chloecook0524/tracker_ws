1. MCTrack 리퍼지토리 설치 & 가상환경 생성
2. conda activate mctrack(가상환경이름)
3. roscore
4. tracker_ws catkin_make 해주기
5. ~/tracker_ws/src/mctrack_tracker/scripts$ python convert_nuscenes_to_results.py 로 valsplit gt json파일생성 
6. ~/tracker_ws$ roslaunch mctrack_tracker run_eval.launch 실행
7. ~/tracker_ws/src/mctrack_tracker/scripts$ python filter_results.py   --input ~/nuscenes_tracking_results.json   --output ~/SOTA/MCTrack/results/nuscenes/mctrack_custom/results.json 트래킹 결과 MCTrack 평가기준에 맞춰 필터링
8. ~/SOTA/MCTrack$ python evaluation/static_evaluation/nuscenes/eval.py results/nuscenes/mctrack_custom/results.json 필터링된 결과를 MCTrack 내부 평가 툴킷으로 성능평가




<현재문제>
mctrack_tracker_node_base.py >  칼만필터만 적용된 베이스 트래커
아래는 베이스 트래커 결과
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
