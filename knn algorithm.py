import math

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.dataset = []
        self.distance_metric = distance_metric

    def fit(self, dataset):
        self.dataset = dataset

    def euclidean_distance(self, point1, point2):
        distance = 0.0
        for i in range(len(point1) - 1):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def manhattan_distance(self, point1, point2):
        distance = 0.0
        for i in range(len(point1) - 1):
            distance += abs(point1[i] - point2[i])
        return distance

    def supremum_distance(self, point1, point2):
        distance = [abs(point1[i] - point2[i]) for i in range(len(point1) - 1)]
        return max(distance)

    def get_distance(self, point1, point2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(point1, point2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(point1, point2)
        elif self.distance_metric == 'supremum':
            return self.supremum_distance(point1, point2)
        else:
            raise ValueError('Please insert a valid distance metric such as euclidean, manhattan or supremum')

    def predict(self, new_point):
        distances = []
        for point in self.dataset:
            dist = self.get_distance(point, new_point)
            distances.append((point, dist))

        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]

        class_votes = {}
        for neighbor in neighbors:
            label = neighbor[0][-1]
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1

        sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]

def calculate_accuracy(test_data, classifier):
    correct_predictions = 0
    for point in test_data:
        predicted_label = classifier.predict(point[:-1])
        actual_label = point[-1]
        if predicted_label == actual_label:
            correct_predictions += 1
    return correct_predictions / len(test_data)

data = [
    [2, 4, 0],
    [4, 6, 0],
    [4, 4, 0],
    [6, 4, 1],
    [6, 6, 1],
    [8, 6, 1]
]

test_data = [
    [3, 5, 0],
    [7, 5, 1],
    [5, 5, 0],
    [6, 7, 1]
]

k = 3

for metric in ['euclidean', 'manhattan', 'supremum']:
    knn_classifier = KNN(k=k, distance_metric=metric)
    knn_classifier.fit(data)
    accuracy = calculate_accuracy(test_data, knn_classifier)
    print(f'The accuracy for the KNN classifier using {metric} distance is {accuracy * 100:.2f}%')
