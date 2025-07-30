import numpy as np
from typing import List, Optional
import random
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Node:
    def __init__(self, point: Point, parent: Optional["Node"]=None):
        self.point = point
        self.parent = parent
        self.cost = 0.0

class Circle:
    def __init__(self, radius: float, center: Point):
        self.radius = radius
        self.center = center

class Rectangle:
    def __init__(self, size: np.ndarray, center: Point, yaw: float):
        self.size = size
        self.center = center
        self.yaw = yaw

class Bound1D:
    def __init__(self, min_range: float, max_range: float):
        self.min_range = min_range
        self.max_range = max_range

class Bound2D:
    def __init__(self, x: Bound1D, y: Bound1D):
        self.x = x
        self.y = y

def get_distance_from_points(a: Point, b: Point):
    return np.linalg.norm([a.x-b.x, a.y-b.y])

class RRT:
    def __init__(self, start_point:Point, end_point: Point, bounds: Bound2D, goal_sample_rate: float = 0.1, step_size: float = 0.01, goal_tolerance: float=1.0):
        self.obstacles: List[ Circle] = []
        self.step_size = step_size
        self.start_point = start_point
        self.end_point = end_point
        self.nodes: list[Node] = [Node(start_point)]
        self.goal_sample_rate = goal_sample_rate
        self.bounds = bounds
        self.goal_tolerance = goal_tolerance

    def add_obstacles(self, obstacles:List[Circle]):
        for obstacle in obstacles:
            self.obstacles.append(obstacle)

    def steer(self, from_point: Point, to_point: Point):
        angle = np.atan2(to_point.y - from_point.y, to_point.x-from_point.x)
        new_point = Point(
            x = self.step_size * np.cos(angle) + from_point.x,
            y = self.step_size * np.sin(angle) + from_point.y
        )
        return new_point
    
    def get_random_point(self) -> Point:
        if random.random() < self.goal_sample_rate:
            return self.end_point
        x = random.randrange(start=self.bounds.x.min_range, stop=self.bounds.x.max_range)
        y = random.randrange(start=self.bounds.y.min_range, stop=self.bounds.y.max_range)
        return Point(x=x, y=y)
    
    
    def get_nearest_node(self, point: Point) -> Node:
        return min(self.nodes, key=lambda x: get_distance_from_points(x.point,point))
    
    def is_collision_free(self, point: Point) -> bool:
        for obs in self.obstacles:
            if np.linalg.norm([obs.center.x-point.x, obs.center.y-point.y])< obs.radius:
                return False
        return True
            
    def extract_path(self):
        path = []
        node = self.nodes[-1]
        while node:
            path.append(node.point)
            node = node.parent
        return path[::-1]

    def draw(self, path: Optional[List[Point]]=None):
        plt.figure()
        for node in self.nodes:
            if node.parent:
                plt.plot([node.point.x, node.parent.point.x],
                         [node.point.y, node.parent.point.y], "-g")

        if path:
            px = [p.x for p in path]
            py = [p.y for p in path]
            plt.plot(px, py, '-r', linewidth=2)

        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle.center.x, obstacle.center.y), obstacle.radius, color='gray')
            plt.gca().add_patch(circle)

        plt.plot(self.start_point.x, self.start_point.y, "bo", label="Start")
        plt.plot(self.end_point.x, self.end_point.y, "ro", label="Goal")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.savefig("rrt_path.png")
        print("Saved plot to rrt_path.png")


    def plan(self, max_iter: float = 10000) -> np.ndarray:
        for _ in range(max_iter):
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(point=random_point)
            new_point = self.steer(from_point=nearest_node.point, to_point=random_point)
            if not self.is_collision_free(point=new_point):
                continue

            new_node = Node(point=new_point, parent=nearest_node)
            self.nodes.append(new_node)
            if get_distance_from_points(a=new_point, b=self.end_point)<self.goal_tolerance:
                self.nodes.append(Node(point=self.end_point, parent=new_node))
                return self.extract_path()
        return None
    
def linearize_points(point_a: Point, point_b: Point, resolution: float = 0.01):
    path = [point_a]
    angle = np.atan2(point_b.x - point_a.x, point_b.y - point_a.y)
    while not get_distance_from_points(path[-1], point_b)<=resolution:
        new_pt = Point(
            x=point_a + np.cos(angle)*resolution,
            y=point_b + np.sin(angle)*resolution
        )
        path.append(new_pt)
    path.append(point_b)
    return path
    
class RRTSTAR(RRT):
    def find_nearby_nodes(self, new_node: Node, radius: float = 1.0):
        nearby_nodes = []
        for node in self.nodes:
            if get_distance_from_points(new_node.point, node.point) <= radius:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def is_collision_free_line(self, from_point: Point, to_point: Point, resolution: float = 0.01):
        # linearize point a to point b
        path = linearize_points(point_a=from_point, point_b=to_point, resolution=resolution)
        for point in path:
            if not self.is_collision_free(point):
                return False
        return True

    def find_best_parent_node(self, new_node: Node, nearby_nodes: list[Node]) -> tuple[Node, float]:
        min_cost = np.inf
        best_parent_node = None
        for node in nearby_nodes:
            cost = node.cost + get_distance_from_points(new_node.point, node.point)
            if min_cost>cost:
                min_cost = cost
                best_parent_node = node
        return best_parent_node, min_cost

    def rewire(self, new_node: Node, nearby_nodes: list[Node]):
        for node in nearby_nodes:
            new_cost = new_node.cost + get_distance_from_points(new_node.point, node.point)
            if node.cost> new_cost:
                node.parent = new_node
                node.cost = new_cost
    
    def plan(self, max_iter = 10000):
        for _ in range(max_iter):
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(point=random_point)
            new_point = self.steer(from_point=nearest_node.point, to_point=random_point)
            if not self.is_collision_free(point=new_point):
                continue
                
            # need to rewire the nodes
            new_node = Node(point=new_point)
            nearby_nodes = self.find_nearby_nodes(new_node=new_node)
            parent_node, cost = self.find_best_parent_node(new_node=new_node, nearby_nodes=nearby_nodes)
            if parent_node is None:
                continue
            new_node.parent = parent_node
            new_node.cost = cost
            self.nodes.append(new_node)
            self.rewire(new_node=new_node, nearby_nodes=nearby_nodes)
            if get_distance_from_points(a=new_point, b=self.end_point)<self.goal_tolerance:
                self.nodes.append(Node(point=self.end_point, parent=new_node))
                return self.extract_path()
        return None
    
if __name__ == "__main__":
    start = Point(0, 0)
    goal = Point(9, 9)
    bounds = Bound2D(
        x=Bound1D(0,10),
        y=Bound1D(0,10)
    )
    goal_sample_rate = 0.05

    rrt = RRTSTAR(start_point=start, end_point=goal, bounds=bounds, goal_sample_rate=goal_sample_rate, step_size=0.5)

    rrt.add_obstacles([Circle(center=Point(5, 5), radius=1.0), Circle(center=Point(3, 6), radius=1.0)])

    path = rrt.plan()
    print(path)
    if path:
        rrt.draw(path=path)