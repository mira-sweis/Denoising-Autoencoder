from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from itertools import permutations
from scipy.optimize import linear_sum_assignment

# put your image generator here
output = decoder(torch.rand(9,4))

def generate_img():
    for i in range(len(output)):
        ax = plt.subplot(3, 3, i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        plt.imshow(output[i].cpu().squeeze().detach().numpy(), cmap='gist_gray')  
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)
    plt.show()
    return

generate_img()


# put your clustering accuracy calculation here

# Function to perform k-means clustering on the encoder outputs
def perform_kmeans(encoder, dataloader, num_clusters):
    encoder_outputs = []

    for image, label in dataloader:
        encode_img = image.to(device)
        encoded_data = encoder(encode_img)
        encoder_outputs.append(encoded_data.cpu().detach().numpy())

    encoder_outputs = np.concatenate(encoder_outputs)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init = 10)
    cluster_assignments = kmeans.fit_predict(encoder_outputs)

    return cluster_assignments

# finds accuracy
def find_accuracy(true_labels, kmeans_labels):
    matrix = np.zeros((10,10))
    index_reassignment = {}
    labels = kmeans_labels

    for i in range(10):
        for j in range(10):
            for k in range(48000):
                if true_labels[k].item() == i and kmeans_labels[k] == j:
                    matrix[i][j] += 1
    
    row,col = linear_sum_assignment(matrix, maximize=True)
    for i, j in zip(row, col):
      index_reassignment[j] = i

    for i in range(0, len(kmeans_labels)):
        labels[i] = index_reassignment[kmeans_labels[i]]
        
    accuracy = accuracy_score(labels, true_labels)

    return accuracy

# Perform k-means clustering on the training set
num_clusters = 10 
train_cluster_assignments = perform_kmeans(encoder, train_loader, num_clusters)

# Get true labels from the training set
true_labels = np.array([label for image, label in train_data])

accuracy = find_accuracy(true_labels, train_cluster_assignments)

print(f"Accuracy: {accuracy}")