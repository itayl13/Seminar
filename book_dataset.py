import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import csv

# from gcmc.data_utils import map_data


def edit_files():
    old_user_f = pd.read_csv(open(os.path.join('data', 'book_crossing_original', 'BX-Users.csv'), 'r'), ';')
    new_user_f = open(os.path.join('data', 'book_crossing_edited', 'BX-Users_new.csv'), 'w')
    w = csv.writer(new_user_f)
    w.writerow(['User-ID', 'Age'])
    valid_user_list = []
    for i in range(old_user_f.shape[0]):
        if np.isnan(old_user_f.loc[i, 'Age']) or 2 > old_user_f.loc[i, 'Age'] or old_user_f.loc[i, 'Age'] > 100:
            continue
        valid_user_list.append(old_user_f.loc[i, 'User-ID'])
        w.writerow([old_user_f.loc[i, 'User-ID'], old_user_f.loc[i, 'Age']])
    new_user_f.close()

    old_book_f = open(os.path.join('data', 'book_crossing_original', 'BX-Books.csv'), 'r')
    new_book_f = open(os.path.join('data', 'book_crossing_edited', 'BX-Books_new_.csv'), 'w')
    wr = csv.writer(new_book_f)
    wr.writerow(['ISBN', 'Book-Author', 'Year-Of-Publication'])
    valid_book_list = []
    for line in old_book_f.readlines():
        spline = line.split(sep=';')
        if len(spline) != 8 or spline[0] == '"ISBN':
            continue
        valid_book_list.append(spline[0].split('"')[1])
        wr.writerow([spline[0].split('"')[1], spline[2].split('""')[1].casefold(), spline[3].split('""')[1]])
    old_book_f.close()
    new_book_f.close()

    valid_books = set(valid_book_list)
    valid_users = set(valid_user_list)
    old_matrix_f = pd.read_csv(open(os.path.join('data', 'book_crossing_original', 'BX-Book-Ratings.csv'), 'r'), ';')
    new_matrix_f = open(os.path.join('data', 'book_crossing_edited', 'BX-Book-Ratings_new.csv'), 'w')
    wrt = csv.writer(new_matrix_f)
    wrt.writerow(['User-ID', 'ISBN', 'Book-Rating'])
    for i in range(old_matrix_f.shape[0]):
        if old_matrix_f.loc[i, 'ISBN'] in valid_books and old_matrix_f.loc[i, 'User-ID'] in valid_users:
            wrt.writerow([old_matrix_f.loc[i, 'User-ID'], old_matrix_f.loc[i, 'ISBN'],
                          old_matrix_f.loc[i, 'Book-Rating']])
    new_matrix_f.close()


def create_book_files(testing):
    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.str,
        'ratings': np.int32}

    matrix_source = pd.read_csv(open(os.path.join('data', 'book_crossing_edited', 'BX-Book-Ratings_new.csv'), 'r'))

    np.random.seed(42)
    test_indices = np.random.choice(range(matrix_source.shape[0]), matrix_source.shape[0] // 10, replace=False)
    train_indices = list(set(range(matrix_source.shape[0])).difference(set(test_indices)))

    data_train = matrix_source[train_indices, :]
    data_test = matrix_source[test_indices, :]

    data_array_train = data_train.as_matrix().tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.as_matrix().tolist()
    data_array_test = np.array(data_array_test)

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.int32)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train+num_val]
    idx_nonzero_test = idx_nonzero[num_train+num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
    pairs_nonzero_test = pairs_nonzero[num_train+num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = range(len(idx_nonzero_train))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    # Side information features
    # book features
    book_df = pd.read_csv(open(os.path.join('data', 'book_crossing_edited', 'BX-Books_new.csv'), 'r'))

    author_dict = {f: i for i, f in enumerate(set(book_df['Book-Author'].values.tolist()), start=2)}
    year = book_df['Year-Of-Publication'].values
    year_max = year.max()

    num_book_feats = 1 + len(author_dict)  # Year of publication (normed), Author (binary by name).

    v_features = np.zeros((num_items, num_book_feats), dtype=np.float32)
    for _, row in book_df.iterrows():
        v_id = row['ISBN']
        # check if book_id was listed in ratings file and therefore in mapping dictionary
        if v_id in v_dict.keys():
            # year
            v_features[v_dict[v_id], 0] = row['Year-Of-Publication'] / np.float(year_max)
            # author
            v_features[v_dict[v_id], author_dict[row['Book-Author']]] = 1.

    # user features
    users_df = pd.read_csv(open(os.path.join('data', 'book_crossing_edited', 'BX-Users_new.csv'), 'r'))

    age = users_df['Age'].values
    age_max = age.max()

    u_features = np.zeros((num_users, 1), dtype=np.float32)
    for _, row in users_df.iterrows():
        u_id = row['user id']
        if u_id in u_dict.keys():
            u_features[u_dict[u_id], 0] = row['Age'] / np.float(age_max)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values
