1. MCTrack 리퍼지토리 설치 & 가상환경 생성
2. conda activate mctrack(가상환경이름)
3. roscore
4. tracker_ws catkin_make 해주기
5. ~/tracker_ws/src/mctrack_tracker/scripts$ python convert_nuscenes_to_results.py 로 valsplit gt json파일생성 
6. ~/tracker_ws$ roslaunch mctrack_tracker run_eval.launch 실행
7. ~/tracker_ws/src/mctrack_tracker/scripts$ python filter_results.py   --input ~/nuscenes_tracking_results.json   --output ~/SOTA/MCTrack/results/nuscenes/mctrack_custom/results.json 트래킹 결과 MCTrack 평가기준에 맞춰 필터링
8. ~/SOTA/MCTrack$ python evaluation/static_evaluation/nuscenes/eval.py results/nuscenes/mctrack_custom/results.json 필터링된 결과를 MCTrack 내부 평가 툴킷으로 성능평가




<현재문제>
mctrack_tracker_node_base.py >  칼만필터만 적용된 기본 트래커로 성능 바닥 상태. 
아래는 해당 트래커 결과
### Final results

| Class     | AMOTA | AMOTP | RECALL | MOTAR | GT    | MOTA  | MOTP  | MT  | ML   | FAF   | TP  | FP   | FN    | IDS  | FRAG | TID   | LGD   |
|-----------|-------|-------|--------|-------|-------|-------|-------|-----|------|--------|-----|------|-------|------|------|--------|--------|
| bicycle   | 0.000 | 2.000 | 0.000  | 0.000 | 1993  | 0.000 | 2.000 | 0   | 156  | 500.0  | 0   | nan  | 1993  | nan  | nan  | 20.00 | 20.00 |
| bus       | 0.000 | 2.000 | 0.000  | 0.000 | 2112  | 0.000 | 2.000 | 0   | 108  | 500.0  | 0   | nan  | 2112  | nan  | nan  | 20.00 | 20.00 |
| car       | 0.000 | 2.000 | 0.000  | 0.000 | 58317 | 0.000 | 2.000 | 0   | 3697 | 500.0  | 0   | nan  | 58317 | nan  | nan  | 20.00 | 20.00 |
| motorcy   | 0.000 | 1.901 | 0.997  | 0.000 | 1977  | 0.000 | 0.012 | 132 | 0    | 36.6   | 259 | 487  | 617   | 12   | 2    | 0.00  | 0.01  |
| pedestr   | 0.000 | 2.000 | 0.000  | 0.000 | 25423 | 0.000 | 2.000 | 0   | 1702 | 500.0  | 0   | nan  | 25423 | nan  | nan  | 20.00 | 20.00 |
| trailer   | 0.000 | 1.950 | 0.998  | 0.000 | 2425  | 0.000 | 0.008 | 133 | 0    | 61.1   | 261 | 613  | 4     | 2160 | 2    | 0.00  | 0.01  |
| truck     | 0.000 | 2.000 | 0.000  | 0.000 | 9650  | 0.000 | 2.000 | 0   | 542  | 500.0  | 0   | nan  | 9650  | nan  | nan  | 20.00 | 20.00 |

**Aggregated Results**  
- **AMOTA**: 0.000  
- **AMOTP**: 1.979  
- **RECALL**: 0.285  
- **MOTA**: 0.000  
- **TP / FP / FN**: 520 / 1100 / 97505  
- **IDS**: 3872  
- **FRAG**: 4  
- **TID / LGD**: 14.29 / 14.29  



mctrack_tracker_node.py 수정하면서 성능 높이면 될 듯
하지만! mctrack 의 핵심특징을 코드에 적용하면 갑자기 검지마저 안되길래 이것저것 단계별로 붙이면서 원인 파악중
