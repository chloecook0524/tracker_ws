1. MCTrack 리퍼지토리 설치 & 가상환경 생성
2. roscore
3. conda activate mctrack(가상환경이름)
4. tracker_ws catkin_make 해주기
5. ~/tracker_ws/src/mctrack_tracker/scripts$ python convert_nuscenes_to_results.py 로 valsplit gt json파일생성 
6. ~/tracker_ws$ roslaunch mctrack_tracker run_eval.launch 실행
7. ~/tracker_ws/src/mctrack_tracker/scripts$ python filter_results.py   --input ~/nuscenes_tracking_results.json   --output ~/SOTA/MCTrack/results/nuscenes/mctrack_custom/results.json 트래킹 결과 mctrack 평가기준에 맞춰 필터링
8. ~/SOTA/MCTrack$ python evaluation/static_evaluation/nuscenes/eval.py results/nuscenes/mctrack_custom/results.json 필터링된 결과를 MCTrack 내부 평가 툴킷으로 성능평가


