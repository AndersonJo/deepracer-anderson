# Training 01 - Negative Reward For Distance-from-center

```python
    def reward_function(...):
        
        if not on_track:
            print('not on track', on_track)
            return -1
        
        print('default reward', -1 * distance_from_center)
        return -1 * distance_from_center
```

![](images/01-result.png)

데모: 

1. 완료
2. 70%
3. 완료
4. 70%
5. 완료
6. 70%
7. 완료

```
Waiting for s3://sagemaker-us-east-1-571175237970/rl-deepracer-sagemaker-190309-151341/output/intermediate/worker_0.simple_rl_graph.main_level.main_level.agent_0.csv...
Downloading rl-deepracer-sagemaker-190309-151341/output/intermediate/worker_0.simple_rl_graph.main_level.main_level.agent_0.csv
```

# Train 02 - Zero Reward

```
    def reward_function(...):
        return 0
```

![](images/02-result.png)



# Train 03

실제 돌려봐도 안돌아감

```python
    def reward_function(...):

        msg = '[Anderson] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, waypoints, closest_waypoints)
        
        if not hasattr(self, '_max_progress'):
            self._max_progress = 0
           
        if self._max_progress < progress:
            print(msg, 'Max Progress')
            self._max_progress = progress
            return progress
        
        if progress >= 100:
            print(msg, 'Done')
            return 100
        
        if not on_track:
            print(msg, 'NOT ON Track')
            return -10
        
        if distance_from_center > 0.05:
            print(msg, 'Distance From Center')
            return -1
        
        print(msg, 'Default')
        return 0
```

![](images/03-result.png)

![](images/03-train.png)



# Train 04 - 일단 됨

돌아감. 1바퀴 돌고.. 반바퀴 더 돌때쯤 멈춤. 일단 됨

문제는.. 속도에 대한 보상이 없었으며 progress가 100이 넘어간후 계속 돌때 계속 1점씩 주기 때문에.. 사실은 게임이 완료 되었지만.. 한바퀴 돌고난 이후에 돌때.. 잘못된 행동에도 리워드가 주어졌음. 

[Video 확인](data/03-validation.mp4)

```python
    def reward_function(...):
        msg = '[Anderson] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:{9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, len(waypoints), closest_waypoints)
        
        if not hasattr(self, '_max_progress'):
            self._max_progress = 0
           
        if self._max_progress < progress:
            print(msg, 'Max Progress')
            self._max_progress = progress
            return 1
        
        if progress >= 100:
            print(msg, 'Done')
            return 1
        
        if not on_track:
            print(msg, 'NOT ON Track')
            return -2
        
        if distance_from_center > 0.05:
            print(msg, 'Distance From Center')
            return -1
        
        print(msg, 'Default')
        return 0
```

![](images/04-result.png)

![](images/04-track.png)

![](images/04-statistic.png)

# Train 05 - Slow Speed & 180 Reward

0.7보다 속도가 떨어질때마다 -1 reward를 주었음. 

또한 progress가 180이상으로 갔을때만 reward 1 값을 주었음. 

잘 안됨. 잘 안돌아감.

```python
    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        
        msg = '[Anderson][04] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:{9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, len(waypoints), closest_waypoints)
        
        if not hasattr(self, '_max_progress'):
            self._max_progress = 0
        
        if not on_track:
            print(msg, 'NOT ON Track')
            return -2
        
        if distance_from_center > 0.05:
            print(msg, 'Distance From Center')
            return -1-distance_from_center
        
        if throttle < 0.7 and progress > 5:
            print(msg, 'Too Slow')
            return -1
        
        if self._max_progress < progress:
            print(msg, 'Max Progress')
            self._max_progress = progress
            return 1
        
        if progress > 180:
            print(msg, 'Progress 180')
            return 1
        
        print(msg, 'Default')
        return 1e-3
```

### Training

![](images/05-result.png)

![](images/05-track.png)

![](images/05-statistic.png)

![](images/05-action.png)

# Train 06 - 개 안됨

자동차는 좌우로 이리저리 흔들리고, 첫번째 커브에서 밖으로 그냥 튀어나감. 

개안됨. 아마도 속도에다가 reward를 줘서 그런가. 확인 필요함

[동영상](data/06-validation-fail.mp4)

```python
    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        
        msg = '[Anderson][04] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:{9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, len(waypoints), closest_waypoints)
        
        if not hasattr(self, '_max_progress'):
            self._max_progress = 0
        
        if not on_track:
            print(msg, 'NOT ON Track')
            return -3
        
        if distance_from_center > 0.05:
            print(msg, 'Distance From Center')
            return - (distance_from_center + 0.5)

        if self._max_progress < progress:
            print(msg, 'Max Progress')
            self._max_progress = progress
            return 1
        
        if progress > 100:
            print(msg, 'progress 100')
            return 1
        
        if throttle < 0.7 and progress > 5:
            print(msg, 'Too Slow')
            return throttle - 1
        
        print(msg, 'Default')
        return 0
```

![](images/06-result.png)

![](images/06-track.png)

![](images/06-statistic.png)

![](images/06-action.png)

# Train 07

초기 eval의 경우 한바퀴는 잘 돌고 그 후 실패하는 경향을 보이다가 추가 eval의 경우 실패

```python
  def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        if distance_from_center >= 0.0 and distance_from_center <= 0.02:
            return 1.0
        elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
            return 0.3
        elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
            return 0.1
        return 1e-3  # like crashed
```

### Training

![](images/07-result.png)

### Evaluation

![](images/07-eval-track.png)

![](images/07-eval-action.png)

# Train 08 - 개안됨

잘 안돌아간다. 

안됨 안됨 안됨 안됨

```python
    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        
        msg = '[Anderson][04] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:{9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, len(waypoints), closest_waypoints)
        
        if not hasattr(self, '_max_progress'):
            print('SET MAX PROGRESS')
            self._max_progress = 0
        
        if not on_track:
            print(msg, 'NOT ON Track -1')
            return -1
        
        if self._max_progress < progress and progress >= 100:
            print(msg, 'Max Progress 1')
            self._max_progress = progress
            return 1
        
        if distance_from_center >= 0.0 and distance_from_center <= 0.02:
            print(msg, 'Good Distance 1.0')
            return 1.0
        elif distance_from_center >= 0.02 and distance_from_center <= 0.03:
            print(msg, 'Good Distance 0.3')
            return 0.3
        elif distance_from_center >= 0.03 and distance_from_center <= 0.05:
            print(msg, 'Good Distance 0.1')
            return 0.1
        
        if progress > 100 and progress < 110:
            print(msg, 'progress 100 ~ 110')
            return 1
        
        print(msg, 'Default', -distance_from_center)
        return -distance_from_center
```

### Training

![](images/08-result.png)

![](images/08-track.png)

![](images/08-statistic.png)

![](images/08-action.png)

### Evaluation

![](images/08-eval-track.png)

![](images/08-eval-statistic.png)

# Train 09 - 인터넷에서 찾은거. 개안됨

참고는 해볼수 있다. 잘 안된다.. 돌기도 하고.. 대부분.. 탈선. 

    def reward_function(self, on_track, x, y, distance_from_center, car_orientation, progress, steps,
                        throttle, steering, track_width, waypoints, closest_waypoints):
        
        msg = '[Anderson][04] on_track:{0} | xy:{1},{2} | dist:{3} | progress:{4} | steps:{5} | throttle:{6} | st:{7} | width:{8} | waypnt:{9} | clswp:{10} | '.format(
               on_track, x, y, round(distance_from_center, 2), round(progress, 2), steps, 
               throttle, steering, track_width, len(waypoints), closest_waypoints)
        
        import math
        from statistics import mean
    
        ##########
        # Settings
        ##########
        # Min / Max Reward
        REWARD_MIN = -1e5
        REWARD_MAX = 1e5
        # Define the Area each side of the center that the card can use.
        # Later version might consider adjust this so that it can hug corners
        CENTER_LANE = track_width * .25
        HALF_TRACK = track_width / 2
    
        ABS_STEERING_THRESHOLD = .85
    
        ####################
        # Locations on track
        ####################
    
        # Set Base Reward
        if not on_track: # Fail them if off Track
            reward = REWARD_MIN
            return reward
        elif progress == 1:
            reward = REWARD_MAX
            return reward
        else:        # we want the vehicle to continue making progress
            reward = REWARD_MAX * progress
    
        # If outside track center than penalize
        if distance_from_center > 0.0 and distance_from_center > CENTER_LANE:
            reward *= 1 - (distance_from_center / HALF_TRACK)
    
        ##########
        # Steering
        ##########
        print('----------------------------------------------------------')
        print('CLOSEST_WAYPOINTS')
        print(closest_waypoints)
    
        # Add penalty for wrong direction
        next_waypoint_yaw = waypoints[min(closest_waypoints+1, len(waypoints)-1)][-1]
        if abs(car_orientation - next_waypoint_yaw) >= math.radians(10):
            reward *= 1 - (abs(car_orientation - next_waypoint_yaw) / 180)
        elif abs(car_orientation - next_waypoint_yaw) < math.radians(10) and abs(steering) > ABS_STEERING_THRESHOLD:    # penalize if stearing to much
            reward *= ABS_STEERING_THRESHOLD / abs(steering)
        else:
            reward *= 1 + (10 - (abs(car_orientation - next_waypoint_yaw) / 10))
    
        # Add penalty if throttle exsides the steering else add reward
        if abs(steering) > .5 and abs(steering > throttle):
            reward *= 1 - (steering - throttle)
        else:
            reward *= 1 + throttle
    
        # make sure reward value returned is within the prescribed value range.
        reward = max(reward, REWARD_MIN)
        reward = min(reward, REWARD_MAX)
    
        return float(reward)
![](images/09-track.png)

