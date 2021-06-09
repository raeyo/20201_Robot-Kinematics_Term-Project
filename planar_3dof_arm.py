"""Reference
https://github.com/AtsushiSakai/PythonRobotics

1.Obstacle navigation using A* on a toroidal grid
Author: Daniel Ingram (daniel-s-ingram)
        Tullio Facchinetti (tullio.facchinetti@unipv.it)

2.Inverse kinematics for an n-link arm using the Jacobian inverse method

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)

3.Inverse kinematics of a two-joint arm
Left-click the plot to set the goal position of the end effector

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)

Ref: P. I. Corke, "Robotics, Vision & Control", Springer 2017,
 ISBN 978-3-319-54413-7 p102
- [Robotics, Vision and Control]
(https://link.springer.com/book/10.1007/978-3-642-20144-8)

"""


from logging import currentframe
from math import pi
from os import link
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import copy
import mpl_toolkits.mplot3d.axes3d as p3
import itertools

from matplotlib.animation import FuncAnimation

from tqdm import tqdm


# Simulation parameters
Kp = 2
dt = 0.1
GOAL_TOLERANCE = 0.1
N_ITERATIONS = 10000
M = 10

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

x = 0
y = 0

# project_info
project_info_msg = """
Planning 3 link arm with A* algorithm
############################################################################
###################  2021 Robot Kinematics Term Project
###################  20204032 Raeyoung Kang

## PARAMETERS
M: num of segments for compute whole configuration space(default: 30)
# main function
"obstacles": list of [x, y, r] for each obstacle
"link_lengths": list of each link length
"joint_angles": list of initial joint angle

## Matplotlib 
Mouse left click to left side plot(Workspace of 3 Link arm): set goal pos
"ESC": Terminate Program(or Ctrl + "C" in terminal) 


############################################################################
"""

class NLinkArm(object):
    """
    Class for controlling and plotting a planar arm with an arbitrary number of links.
    """

    def __init__(self, link_lengths, joint_angles, obstacles=[]):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.obstacles = obstacles
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

        self.collision_config = [[],[],[]] #TODO: only available for 3dof arm
        self.free_config = [[],[],[]]

        self.workspace_path = []
        self.configspace_path = []

        self.initialize_fig()

    def update_joints(self, joint_angles):
        joint_angles = np.array(joint_angles)
        joint_angles = (joint_angles + np.pi) % (2 * np.pi) - np.pi
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T

    def check_collision_within_link(self):
        # check collision among links
        def get_line(line_points):
            # y = ax + b
            point1 = line_points[0]
            point2 = line_points[1]
            if point1[0] == point2[0]:
                return False
            else:
                a = (point2[1] - point1[1])/(point2[0] - point1[0])
                b = point1[1] - a*point1[0]
                x_range = [min(point1[0], point2[0]), max(point1[0], point2[0])]
                return (a, b, x_range)
            
        line_segs = [[self.points[i], self.points[i+1]]for i in range(len(self.points) - 1)]
        is_collision = False
        for i in range(len(line_segs) - 1):
            line1 = get_line(line_segs[i])
            line2 = get_line(line_segs[i+1])
            if type(line1)==tuple and type(line2)==tuple:
                a1, b1, x_range1 = line1
                a2, b2, x_range2 = line2
                if abs(a1 - a2) < 1e-20:
                    if abs(b1 - b2) < 1e-20:
                        is_collision = True
                    else:
                        is_collision = False
                else:
                    intersect_x = (b2 - b1) / (a1 - a2)
                    intersect_range = [min(x_range1[0], x_range2[0]), max(x_range1[1], x_range2[1])]
                    if intersect_range[0] > intersect_range[1]:
                        is_collision = False
                    else:
                        if intersect_range[0] < intersect_x < intersect_range[1]:
                            is_collision = True
                        else:
                            is_collision = False
            elif type(line1)==bool and type(line2)==bool:
                point1, point2 = line_segs[i]
                point3, point4 = line_segs[i+1]
                if point1[0] == point3[0]:
                    y_range1 = [min(point1[1], point2[1]), max(point1[1], point2[1])]
                    y_range2 = [min(point3[1], point4[1]), max(point3[1], point4[1])]
                    y_intersect_range = [min(y_range1[0], y_range2[0]), max(y_range1[1], y_range2[1])]
                    if y_intersect_range[0] > y_intersect_range[1]:
                        is_collision = False
                    else:
                        is_collision = True
                else:
                    is_collision = False                
            else:
                point1, point2 = line_segs[i]
                point3, point4 = line_segs[i+1]
                if type(line1)==bool:
                    a2, b2, x_range2 = line2
                    if x_range2[0] < point1[0] < x_range2[1]:
                        y_intersect_point = a2 * point1[0] + b2
                        y_range1 = [min(point1[1], point2[1]), max(point1[1], point2[1])]
                        if y_range1[0] < y_intersect_point < y_range1[1]:
                            is_collision = True
                        else:
                            is_collision = False
                    else:
                        is_collision = False
                else:
                    a1, b1, x_range1 = line1
                    if x_range1[0] < point3[0] < x_range1[1]:
                        y_intersect_point = a1 * point3[0] + b1
                        y_range2 = [min(point3[1], point4[1]), max(point3[1], point4[1])]
                        if y_range2[0] < y_intersect_point < y_range2[1]:
                            is_collision = True
                        else:
                            is_collision = False
                    else:
                        is_collision = False

            if is_collision:
                return is_collision
        return is_collision

    def check_collision_with_obstacle(self):
        def detect_collision(line_seg, circle):
            """
            Determines whether a line segment (arm link) is in contact
            with a circle (obstacle).
            Credit to: http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html
            Args:
                line_seg: List of coordinates of line segment endpoints e.g. [[1, 1], [2, 2]]
                circle: List of circle coordinates and radius e.g. [0, 0, 0.5] is a circle centered
                        at the origin with radius 0.5

            Returns:
                True if the line segment is in contact with the circle
                False otherwise
            """
            a_vec = np.array([line_seg[0][0], line_seg[0][1]])
            b_vec = np.array([line_seg[1][0], line_seg[1][1]])
            c_vec = np.array([circle[0], circle[1]])
            radius = circle[2]
            line_vec = b_vec - a_vec
            line_mag = np.linalg.norm(line_vec)
            circle_vec = c_vec - a_vec
            proj = circle_vec.dot(line_vec / line_mag)
            if proj <= 0:
                closest_point = a_vec
            elif proj >= line_mag:
                closest_point = b_vec
            else:
                closest_point = a_vec + line_vec * proj / line_mag
            if np.linalg.norm(closest_point - c_vec) > radius:
                return False

            return True
        collision_detected = False
        for k in range(len(self.points) - 1):
            for obstacle in self.obstacles:
                line_seg = [self.points[k], self.points[k + 1]]
                collision_detected = detect_collision(line_seg, obstacle)
                if collision_detected:
                    break
            if collision_detected:
                    break
        return collision_detected

    def initialize_fig(self):
        self.fig = plt.figure(figsize=(20 ,10))

        self.fig.canvas.mpl_connect("button_press_event", self.click)
        # for stopping simulation with the esc key.
        self.fig.canvas.mpl_connect('key_release_event', lambda event: [
                            exit(0) if event.key == 'escape' else None])
        

        #1 plot for arm
        self.workspace_fig = self.fig.add_subplot(1, 2, 1)
        self.workspace_fig.set_xlim([-self.lim, self.lim])
        self.workspace_fig.set_ylim([-self.lim, self.lim])
        
        #1.1 draw obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(
                (obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            self.workspace_fig.add_patch(circle)
        self.link_lines = []

        #1.2 draw links
        for i in range(self.n_links + 1):
            if i is not self.n_links:
                line = self.workspace_fig.plot([self.points[i][0], self.points[i + 1][0]],
                                [self.points[i][1], self.points[i + 1][1]], 'r-',
                                 self.points[i][0], self.points[i][1], 'k.')
            else:
                line = self.workspace_fig.plot(self.points[i][0], self.points[i][1], 'k.')
            self.link_lines.append(line)
        
        #1.3 draw end effector path
        self.workspace_path_fig = self.workspace_fig.plot([], [], 'g*')[0]
        #1.4 draw target goal
        self.workspace_target_fig = self.workspace_fig.plot([], [], marker='D', color='g')[0]


        #2 plot for configuration space
        self.config_space_fig = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.config_space_fig.set_xlim([-np.pi, np.pi])
        self.config_space_fig.set_ylim([-np.pi, np.pi])
        self.config_space_fig.set_zlim([-np.pi, np.pi])
        self.config_space_fig.set_xlabel("theta1")
        self.config_space_fig.set_ylabel("theta2")
        self.config_space_fig.set_zlabel("theta3")

        #2.1 draw collision space
        self.collision_space_graph = self.config_space_fig.scatter(self.collision_config[0],
                                                                   self.collision_config[1],
                                                                   self.collision_config[2], color="k", alpha=0.1)
        #2.2 draw free space
        self.free_space_graph = self.config_space_fig.scatter(self.free_config[0],
                                                              self.free_config[1],
                                                              self.free_config[2], color="c")
        #2.3 draw config space path
        self.config_space_path_fig = self.config_space_fig.plot([],[],[],"g*")[0]
        #2.4 draw target joint goal
        self.config_space_target_fig = self.config_space_fig.plot([], [], [], marker='D', color='g')[0]
        #2.5 draw current joint angle
        self.config_space_current_fig = self.config_space_fig.plot([],
                                                                   [],
                                                                   [],
                                                                   marker="o",

                                                                   color="r")[0]

    def plot(self):
        self.update_arm_plot()
        self.update_config_plot()
        
        plt.show()
        plt.pause(dt)
    
    def update_arm_plot(self):  # pragma: no cover
        for i in range(self.n_links + 1):
            if i is not self.n_links:
                self.link_lines[i][0].set_data([self.points[i][0], self.points[i + 1][0]],
                                            [self.points[i][1], self.points[i + 1][1]])
                self.link_lines[i][1].set_data(self.points[i][0], self.points[i][1])
            else:
                self.link_lines[i][0].set_data(self.points[i][0], self.points[i][1])

    def update_config_plot(self):
        # update graph
        self.collision_space_graph._offsets3d = (self.collision_config[0],
                                                 self.collision_config[1],
                                                 self.collision_config[2])
        # self.free_space_graph._offsets3d = (self.free_config[0],
        #                                     self.free_config[1],
        #                                     self.free_config[2])
        self.config_space_current_fig.set_data(self.joint_angles[:2])
        self.config_space_current_fig.set_3d_properties(self.joint_angles[2])

    def compute_configspace(self, segments=10, animate=False, plotly=False):
        theta1_space = np.linspace(-np.pi, np.pi, segments)
        theta2_space = np.linspace(-np.pi, np.pi, segments)
        theta3_space = np.linspace(-np.pi, np.pi, segments)

        self.config_space = []
        self.config_grid = []    
        theta_space = itertools.product(theta1_space, theta2_space, theta3_space)
        
        if animate:
            ani = FuncAnimation(self.fig, self.update_config_with_animate, frames=theta_space, interval=40)
            input("Press Enter to show animation")
            plt.show()    
        else:
            for theta in tqdm(theta_space):
                self.update_config(theta)
        self.config_space = np.array(self.config_space).reshape((M,M,M,-1))
        
        if plotly:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Scatter3d(x=self.collision_config[0], 
                                               y=self.collision_config[1], 
                                               z=self.collision_config[2],
                                               text=("theta1",
                                                      "theta2",
                                                      "theta3"),
                                    mode='markers',
                                    marker=dict(opacity=0.2))])

            fig.show()
    
    def compute_configspace_incremantal(self, segments=10, animate=False):
        for i in range(5):
            if i == 0:
                theta1_space = np.linspace(-np.pi, np.pi, segments)
                theta2_space = np.linspace(-np.pi, np.pi, segments)
                theta3_space = np.linspace(-np.pi, np.pi, segments)
                self.config_space = []
                self.config_grid = []    
                
                theta_space = itertools.product(theta1_space, theta2_space, theta3_space)
            else:
                previous_collision_cofig = np.array(self.collision_config)
                random_noise = np.random.uniform(-np.pi, np.pi)/segments
                previous_collision_cofig = previous_collision_cofig + random_noise
                previous_collision_cofig = np.clip(previous_collision_cofig, -np.pi, np.pi)
                theta_space = [(previous_collision_cofig[0,j],previous_collision_cofig[1,j],previous_collision_cofig[2,j]) \
                    for j in range(len(previous_collision_cofig[0]))]
            
            if animate:
                ani = FuncAnimation(self.fig, self.update_config_with_animate, frames=theta_space, interval=40)
                plt.show()    
            else:
                for theta in tqdm(theta_space):
                    self.update_config(theta)
            
                import plotly.graph_objects as go
                
                
                fig = go.Figure(data=[go.Scatter3d(x=self.collision_config[0], 
                                                   y=self.collision_config[1], 
                                                   z=self.collision_config[2],
                                                   text=("theta1",
                                                   "theta2",
                                                   "theta3"),
                                    mode='markers',
                                    marker=dict(opacity=0.2))])
                fig.show()

    def update_config(self, joint_angles):
        self.update_joints(joint_angles)
        self.config_space.append(np.array(joint_angles))
        if self.check_collision_with_obstacle():
            for i in range(self.n_links):
                self.collision_config[i].append(joint_angles[i])
            self.config_grid.append(1)
        else:
            for i in range(self.n_links):
                self.free_config[i].append(joint_angles[i])
            self.config_grid.append(0)
    
    def update_config_with_animate(self, joint_angles):
        self.update_config(joint_angles)
        # for n link arm
        self.update_arm_plot()
        # for configuratin space
        self.update_config_plot()

    def inverse_kinematics(self, goal_pos):
        """
        Calculates the inverse kinematics using the Jacobian inverse method.
        """
        joint_angles = self.joint_angles
        for iteration in range(N_ITERATIONS):
            current_pos = self.forward_kinematics(joint_angles)
            errors, distance = distance_to_goal(current_pos, goal_pos)
            if distance < GOAL_TOLERANCE:
                print("Solution found in %d iterations." % iteration)
                joint_angles = (joint_angles + np.pi) % (2 * np.pi) - np.pi
                return joint_angles, True
            J = self.jacobian_inverse(joint_angles)
            joint_angles = joint_angles + np.matmul(J, errors)
        
        return joint_angles, False

    def jacobian_inverse(self, joint_angles):
        J = np.zeros((2, self.n_links))
        for i in range(self.n_links):
            J[0, i] = 0
            J[1, i] = 0
            for j in range(i, self.n_links):
                J[0, i] -= self.link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
                J[1, i] += self.link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

        return np.linalg.pinv(J)

    def forward_kinematics(self, joint_angles):
        x = y = 0
        for i in range(1, self.n_links + 1):
            x += self.link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
            y += self.link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
        return np.array([x, y]).T

    def animation(self):
        global x, y
        x_prev, y_prev = None, None

        state = WAIT_FOR_NEW_GOAL
        route = []
        route_idx = 0
        while True:
            try:
                if x is not None and y is not None:
                    x_prev = x
                    y_prev = y
                
                if state is WAIT_FOR_NEW_GOAL:
                    goal_pos = np.array([x, y])
                
                    errors, distance = distance_to_goal(self.end_effector, goal_pos)
                    if distance > GOAL_TOLERANCE:
                        self.workspace_target_fig.set_data(goal_pos)
                        
                        if np.linalg.norm(goal_pos) > 3:
                            print("unreachable taget {}, {}".format(x, y))
                            print("Try other target")
                        else:
                            print("Try to solve IK")
                            goal_joint_angles, solution_found = self.inverse_kinematics(goal_pos)
                            if solution_found:
                                print("Success to solve IK")
                                self.config_space_target_fig.set_data(goal_joint_angles[:2])
                                self.config_space_target_fig.set_3d_properties(goal_joint_angles[2])
                                print("Find Path to Goal: {}".format(goal_joint_angles))
                                
                                route, _, _  = self.astar_tour(self.joint_angles, goal_joint_angles)
                                state = MOVING_TO_GOAL

                            else:
                                print("Fail to Solve IK")
                                print("Try other target")
                elif state is MOVING_TO_GOAL:
                    errors, distance = distance_to_goal(self.end_effector, goal_pos)
                    if distance > GOAL_TOLERANCE and (len(route)-1) == route_idx:
                        joint_angles = self.joint_angles + Kp * \
                            ang_diff(goal_joint_angles, self.joint_angles) * dt
                        self.update_joints(joint_angles)
                    
                    elif distance > GOAL_TOLERANCE and (len(route)-1) > route_idx:
                        current_goal_joint = self.config_space[route[route_idx]]
                        if np.linalg.norm(ang_diff(current_goal_joint, self.joint_angles)) > GOAL_TOLERANCE:
                            joint_angles = self.joint_angles + Kp * \
                                ang_diff(current_goal_joint, self.joint_angles) * 0.5
                            self.update_joints(joint_angles)
                        else:
                            route_idx += 1
                            current_goal_joint = self.config_space[route[route_idx]]
                            joint_angles = self.joint_angles + Kp * \
                                ang_diff(current_goal_joint, self.joint_angles) * 0.5
                            self.update_joints(joint_angles)
                    else:
                        goal_joint_angles = []
                        state = WAIT_FOR_NEW_GOAL
                        route = []
                        route_idx = 0
                        solution_found = False

                self.plot()
                                
            except ValueError as e:
                print("Unreachable goal"+e)
            except TypeError:
                x = x_prev
                y = y_prev

    def animation_with_collision(self, iter_goal=5):
        global x, y
        x_prev, y_prev = None, None

        state = WAIT_FOR_NEW_GOAL
        route = []
        route_idx = 0
        while True:
            try:
                if x is not None and y is not None:
                    x_prev = x
                    y_prev = y
                
                if state is WAIT_FOR_NEW_GOAL:
                    goal_pos = np.array([x, y])
                
                    errors, distance = distance_to_goal(self.end_effector, goal_pos)
                    if distance > GOAL_TOLERANCE:
                        self.workspace_target_fig.set_data(goal_pos)
                        
                        if np.linalg.norm(goal_pos) > 3:
                            print("unreachable taget {}, {}".format(x, y))
                            print("Try other target")
                        else:
                            print("Try to solve IK")
                            goal_joint_angles, solution_found = self.inverse_kinematics(goal_pos)
                            if solution_found:
                                print("Success to solve IK")
                                self.config_space_target_fig.set_data(goal_joint_angles[:2])
                                self.config_space_target_fig.set_3d_properties(goal_joint_angles[2])
                                joint_angles = self.joint_angles + Kp * \
                                ang_diff(goal_joint_angles, self.joint_angles) * dt
                                self.update_joints(joint_angles)
                                state = MOVING_TO_GOAL
                            else:
                                print("Fail to Solve IK")
                                print("Try other target")
                elif state is MOVING_TO_GOAL:
                    errors, distance = distance_to_goal(self.end_effector, goal_pos)
                    if distance > GOAL_TOLERANCE:
                        joint_angles = self.joint_angles + Kp * \
                            ang_diff(goal_joint_angles, self.joint_angles) * dt
                        self.update_joints(joint_angles)
                    else:
                        goal_joint_angles = []
                        state = WAIT_FOR_NEW_GOAL
                        solution_found = False

                self.plot()
                                

            except ValueError as e:
                print("Unreachable goal"+e)
            except TypeError:
                x = x_prev
                y = y_prev
    
    def click(self, event):  # pragma: no cover
        global x, y
        print("click")
        if self.workspace_fig == event.inaxes:
            x = event.xdata
            y = event.ydata

    def astar_tour(self, start_state=[0,0,0], goal_state=[np.pi,np.pi,np.pi]):
        colors = ['white', 'black', 'red', 'pink', 'yellow', 'green', 'orange']
        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        cmap, norm = from_levels_and_colors(levels, colors)
        
        config_space = np.array(self.config_space).reshape((M,M,M,-1))
        config_grid = np.array(self.config_grid).reshape((M, M, M))

        # get closest node of start and goal
        dif_start = config_space - start_state
        dif_start = np.linalg.norm(dif_start, axis=3)
        start_node = np.unravel_index(np.argmin(dif_start, axis=None), dif_start.shape)

        dif_goal = config_space - goal_state
        dif_goal = np.linalg.norm(dif_goal, axis=3)
        goal_node = np.unravel_index(np.argmin(dif_goal, axis=None), dif_goal.shape)

        config_grid[start_node] = 4
        config_grid[goal_node] = 5

        parent_map =[[[() for _ in range(M)] for _ in range(M)] for _ in range(M)]
        heuristic_map = calcheuristic(goal_node)

        explored_heuristic_map = np.full((M, M, M), np.inf)
        distance_map = np.full((M, M, M), np.inf)
        explored_heuristic_map[start_node] = heuristic_map[start_node]
        distance_map[start_node] = 0

        while True:
            config_grid[start_node] = 4
            config_grid[goal_node] = 5
            
            current_node = np.unravel_index(
            np.argmin(explored_heuristic_map, axis=None), explored_heuristic_map.shape)
            min_distance = np.min(explored_heuristic_map)
            if (current_node == goal_node) or np.isinf(min_distance):
                break
            
            config_grid[current_node] = 2

            explored_heuristic_map[current_node] = np.inf
            i, j, k = current_node[0], current_node[1], current_node[2]

            neighbors = find_neighbors(i, j, k)
            for neighbor in neighbors:
                if config_grid[neighbor] == 0 or config_grid[neighbor] == 5:
                    distance_map[neighbor] = distance_map[current_node] + 1
                    explored_heuristic_map[neighbor] = heuristic_map[neighbor]
                    parent_map[neighbor[0]][neighbor[1]][neighbor[2]] = current_node
                    config_grid[neighbor] = 3

        if np.isinf(explored_heuristic_map[goal_node]):
            route = []
            print("No route found.")
        else:
            route = [goal_node]
            while parent_map[route[0][0]][route[0][1]][route[0][2]] != ():
                route.insert(0, parent_map[route[0][0]][route[0][1]][route[0][2]])

            print("The route found covers %d grid cells." % len(route))

        return route, cmap, norm
        
def calcheuristic(goal):
    heuristic = np.zeros((M,M,M))
    for z in range(M):
        for y in range(M):
            for x in range(M):
                 
                # Euklidische Distanz für jede Zelle zum Ziel berechnen
                dist=((x-goal[0])**2+(y-goal[1])**2+(z-goal[2])**2)**(1/2.0)
            
                # Höhe
                zheu = -6.0*float(z)
                
                # Horizontale von Soll
                yheu = np.abs(float(y) - goal[1])
                
                # und Höhe und Abweichung von y=0
                heuristic[x,y,z]= dist + yheu #+ zheu
    '''     
    for i in range(len(heuristic)):
        print(heuristic[i])
    '''
    return heuristic

def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)

def ang_diff(theta1, theta2):
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

def find_neighbors(i, j, k):
    neighbors = []
    if i - 1 >= 0:
        neighbors.append((i - 1, j, k))
    else:
        neighbors.append((M - 1, j, k))

    if i + 1 < M:
        neighbors.append((i + 1, j, k))
    else:
        neighbors.append((0, j, k))

    if j - 1 >= 0:
        neighbors.append((i, j - 1, k))
    else:
        neighbors.append((i, M - 1, k))

    if j + 1 < M:
        neighbors.append((i, j + 1, k))
    else:
        neighbors.append((i, 0, k))

    if k - 1 >= 0:
        neighbors.append((i, j , k- 1))
    else:
        neighbors.append((i, j , M - 1))

    if k + 1 < M:
        neighbors.append((i, j, k + 1))
    else:
        neighbors.append((i, j, 0))


    return neighbors

def main():
    global M, x
    obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25]] # x, y, r
    link_lenghts = [1, 1, 1]
    joint_angles = [0, 0, 0]
    
    arm = NLinkArm(link_lengths=link_lenghts,
                   joint_angles=joint_angles,
                   obstacles=obstacles)
    
    # compute config space
    arm.compute_configspace(segments=M, animate=False, plotly=False)
    # arm.compute_configspace_incremantal(segments=20, animate=False)
    
    # initialize to zero
    x = sum(link_lenghts)
    plt.ion()
    arm.update_joints(joint_angles)
    arm.plot()

    arm.animation()

if __name__=="__main__":
    print(project_info_msg)
    main()