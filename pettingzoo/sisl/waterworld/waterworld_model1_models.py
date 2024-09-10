import numpy as np
import pygame
import pymunk
from gymnasium import spaces


class Obstacle:
    def __init__(self, x, y, pixel_scale=750, radius=0.1):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.body.position = x, y
        self.body.velocity = 0.0, 0.0

        self.shape = pymunk.Circle(self.body, pixel_scale * 0.1)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.custom_value = 1

        self.radius = radius * pixel_scale
        self.color = (120, 176, 178)

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class MovingObject:
    def __init__(self, x, y, pixel_scale=750, radius=0.015):
        self.pixel_scale = 30 * 25
        self.body = pymunk.Body()
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity

        self.radius = radius * pixel_scale

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy


class Evaders(MovingObject):
    def __init__(self, x, y, vx, vy, radius=0.03, collision_type=2, max_speed=100, nutrition=2):
        super().__init__(x, y, radius=radius)

        self.body.velocity = vx, vy

        self.color = (145, 250, 116)
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.max_speed = max_speed
        self.shape.density = 0.01
        self.nutrition=nutrition
        self.eaten=False

    def reset(self, x, y, vx, vy):
        self.body.position = x, y
        self.body.velocity = vx, vy
        self.eaten = False  # Reset eaten state

class Poisons(MovingObject):
    def __init__(
        self, x, y, vx, vy, radius=0.015 * 3 / 4, collision_type=3, max_speed=100
    ):
        super().__init__(x, y, radius=radius)

        self.body.velocity = vx, vy

        self.color = (238, 116, 106)
        self.shape.collision_type = collision_type
        self.shape.max_speed = max_speed


class Pursuers(MovingObject):
    def __init__(
        self,
        x,
        y,
        max_accel,
        pursuer_speed,
        radius=0.015,
        n_sensors=30,
        sensor_range=0.2,
        collision_type=1,
        speed_features=True,
        max_satiety = 5,
        initial_satiety=0.5,
        initial_arousal=0.5,
        satiety_decay_rate=0.01,
        arousal_decay_rate=0.005,
        haptic_modulation_type="average",
        satiety_arousal_rate=0,
        haptic_weight=0.5,
        max_sensor_range=1
    ):  
        super().__init__(x, y, radius=radius)

        self.color = (101, 104, 249)
        self.shape.collision_type = collision_type
        self.sensor_color = (0, 0, 0)
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range * self.pixel_scale
        self.max_accel = max_accel
        self.max_speed = pursuer_speed
        self.body.velocity = 0.0, 0.0

        self.satiety = initial_satiety
        self.arousal = initial_arousal
        self.max_satiety = max_satiety
        self.satiety_decay_rate = satiety_decay_rate
        self.arousal_decay_rate = arousal_decay_rate
        self.haptic_modulation_type = haptic_modulation_type
        self.satiety_arousal_rate = satiety_arousal_rate
        self.social_haptic_modulation = 0
        self.haptic_weight = haptic_weight
        
        self.shape.food_indicator = 0  # 1 if food caught at this step, 0 otherwise
        self.shape.food_touched_indicator = (
            0  # 1 if food touched as this step, 0 otherwise
        )
        self.shape.poison_indicator = 0  # 1 if poisoned this step, 0 otherwise
        self.shape.social_touch_indicator = 0

        # Generate self.n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors + 1)[:-1]

        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors
        self.shape.custom_value = 1

        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 5
        if speed_features:
            self._sensor_obscoord += 3

        self.sensor_obs_coord = self.n_sensors * self._sensor_obscoord
        self.obs_dim = (
            self.sensor_obs_coord + 2
        )  # +1 for is_colliding_evader, +1 for is_colliding_poison
    
    def update_awareness(self, dt, other_pursuers, evaders):

        # Calculate arousal based on nearby pursuers and social haptic modulation due to these contacts
        #first initialise social haptic modulation value as 0
        pursuers_in_contact = [self]
        total_arousal = self.arousal
        max_arousal = self.arousal
        min_arousal = self.arousal

        for other in other_pursuers:
            #not a very good implementation right now, only checking if there was a social touch, this works only in two 
            #agent scenarios - BEWARE, TODO need to have an indicator for which was the other agent, couldnt implement it bc
            #i dont have access to the indices from the shape of object. Probably worst case the commented out code below works
            #but want to keep to the original code convention of using callback and touch indicators
            if other != self and self.shape.social_touch_indicator >0 :
            #self.distance_to(other) < self.radius + other.radius:
                pursuers_in_contact.append(other)
                total_arousal += other.arousal
                max_arousal = max(max_arousal, other.arousal)
                min_arousal = min(min_arousal, other.arousal)
        if len(pursuers_in_contact) > 1:
            if self.haptic_modulation_type == "average":
                average_arousal = total_arousal / len(pursuers_in_contact)
                self.social_haptic_modulation = (average_arousal - self.arousal) * self.haptic_weight
            elif self.haptic_modulation_type == "cooperative":
                self.social_haptic_modulation = (max_arousal - self.arousal) * self.haptic_weight
            elif self.haptic_modulation_type == "competitive":
                self.social_haptic_modulation = (min_arousal - self.arousal) * self.haptic_weight
            elif self.haptic_modulation_type == "no_effect":
                self.social_haptic_modulation = 0
        else:
            self.social_haptic_modulation = 0


        #Update arouusal influenced both by hunger and arousal modulation coming from haptic contact
        hunger = self.max_satiety - self.satiety
        self.arousal += hunger * 0.01 * dt + self.social_haptic_modulation
        self.arousal = np.clip(self.arousal, -1, 1)

         # Update satiety and radius
        satiety_decrease_rate = self.satiety_decay_rate * (1 + self.arousal)
        self.satiety = max(self.satiety - satiety_decrease_rate * dt, 0)
        
        # Decay and clip arousal
        if self.arousal > 0:
            self.arousal = max(0, self.arousal - self.arousal_decay_rate * dt)
        else:
            self.arousal = min(0, self.arousal + self.arousal_decay_rate * dt)
        
        #clip to be in certain range
        self.arousal = np.clip(self.arousal, -1, 1)

        # Update sensor range
        new_sensor_range = self.sensor_range * (self.initial_arousal + self.arousal)
        self.sensor_range = min(new_sensor_range, self.max_sensor_range)

    def distance_to(self, other):
        return np.linalg.norm(np.array(self.body.position) - np.array(other.body.position))
    
    # Eating behaviour here to update satiety
    def eat(self, food_nutrition):
        previous_satiety = self.satiety
        self.satiety = min(self.max_satiety, previous_satiety + food_nutrition)
        self.arousal += self.satiety_arousal_rate 
        self.arousal = np.clip(self.arousal, -1, 1)

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-2 * np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.Box(
            low=np.float32(-self.max_accel),
            high=np.float32(self.max_accel),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def position(self):
        assert self.body.position is not None
        return np.array([self.body.position[0], self.body.position[1]])

    @property
    def velocity(self):
        assert self.body.velocity is not None
        return np.array([self.body.velocity[0], self.body.velocity[1]])

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def draw(self, display, convert_coordinates):
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self.sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_barrier_readings(self):
        """Get the distance to the barrier.

        See https://github.com/BolunDai0216/WaterworldRevamp for
        a detailed explanation.
        """
        # Get the endpoint position of each sensor
        sensor_vectors = self._sensors * self.sensor_range
        position_vec = np.array([self.body.position.x, self.body.position.y])
        sensor_endpoints = position_vec + sensor_vectors

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, 0.0, self.pixel_scale)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - position_vec

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio will limit the end of the vector to the barriers
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=np.abs(sensor_vectors) > 1e-8,
        )

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to 1.0
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = 1.0

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0

        return sensor_values[0, :]

    def get_sensor_reading(
        self, object_coord, object_radius, object_velocity, object_max_velocity
    ):
        """Get distance and velocity to another object (Obstacle, Pursuer, Evader, Poison)."""
        # Get location and velocity of pursuer
        self.center = self.body.position
        _velocity = self.body.velocity

        # Get distance of object in local frame as a 2x1 numpy array
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # Get relative velocity as a 2x1 numpy array
        relative_speed = np.array(
            [
                [object_velocity[0] - _velocity[0]],
                [object_velocity[1] - _velocity[1]],
            ]
        )

        # Project distance to sensor vectors
        sensor_distances = self._sensors @ distance_vec

        # Project velocity vector to sensor vectors
        sensor_velocities = (
            self._sensors @ relative_speed / (object_max_velocity + self.max_speed)
        )

        # if np.any(sensor_velocities < -2 * np.sqrt(2)) or np.any(
        #     sensor_velocities > 2 * np.sqrt(2)
        # ):
        #     set_trace()

        # Check for valid detection criterions
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self.sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # Set not sensed sensor readings of position to sensor range
        sensor_distances = np.clip(sensor_distances / self.sensor_range, 0, 1)
        sensor_distances[not_sensed_idx] = 1.0

        # Set not sensed sensor readings of velocity to zero
        sensor_velocities[not_sensed_idx] = 0.0

        return sensor_distances, sensor_velocities
