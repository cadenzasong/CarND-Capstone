import random
import copy


class LabeledPoint(object):
    """
    label: anything
    point: (number, number)
    """

    def __init__(self, label, point):
        self.label = label
        self.point = point

    def dist_square(self, point):
        return (point[0] - self.point[0]) * (point[0] - self.point[0]) + (point[1] - self.point[1]) * (point[1] - self.point[1])


class TwoDTree(object):
    """
    items: list of LabeledPoint's
    """

    def __init__(self, items, depth=0):
        self.axis = depth % 2
        self.left = None
        self.right = None
        self.item = None
        self.tmp_other = False

        if len(items) == 0:
            self.item = None
            return

        if len(items) == 1:
            self.item = items[0]
            return

        splitter = self.randomized_median_approximation(items, self.axis)
        self.item = splitter

        left_items = []
        right_items = []

        for item in items:
            if item != splitter:
                if item.point[self.axis] < splitter.point[self.axis]:
                    left_items.append(item)
                else:
                    right_items.append(item)
        if left_items:
            self.left = TwoDTree(left_items, depth + 1)
        if right_items:
            self.right = TwoDTree(right_items, depth + 1)

    @staticmethod
    def randomized_median_approximation(items, axis):
        sample_size = 10
        if len(items) < sample_size:
            sample = copy.copy(items)
        else:
            sample = random.sample(items, sample_size)

        sample.sort(key=lambda item: item.point[axis])
        return sample[len(sample) // 2]

    def find_closest(self, point):
        if self.item == None:  # empty tree
            return None

        down = True
        closest = None
        closest_dist_square = 1e9
        stack = [self]
        while stack:
            curr = stack[-1]
            if down:
                if point[0] == curr.item.point[0] and point[1] == curr.item.point[1]:
                    return curr.item
                if point[curr.axis] < curr.item.point[curr.axis]:
                    next_ = curr.left
                    curr.tmp_other = curr.right
                else:
                    next_ = curr.right
                    curr.tmp_other = curr.left
                if next_ != None:
                    stack.append(next_)
                else:
                    dist_square = curr.item.dist_square(point)
                    if dist_square < closest_dist_square:
                        closest = curr
                        closest_dist_square = dist_square
                    down = False
            else:  # up
                stack.pop()  # we don't need it anymore
                dist_square = curr.item.dist_square(point)
                if dist_square < closest_dist_square:
                    closest = curr
                    closest_dist_square = dist_square
                if curr.tmp_other != None and curr.dist_square_on_its_axis(point) < closest_dist_square:
                    stack.append(curr.tmp_other)
                    down = True
        return closest.item if closest else None

    def dist_square_on_its_axis(self, point):
        signed_dist = point[self.axis] - self.item.point[self.axis]
        return signed_dist * signed_dist
