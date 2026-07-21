from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from src.utils import save_model

#Train on original set only
def train_logistic_regression(X_train, y_train, class_weight = "balanced", random_state=42):
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    save_model(model, "logistic_regression")
    return model

#Train model on Synthetic data
def train_logistic_regression_sm(X_train, y_train, random_state=42):
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, class_weight="balanced", random_state=42):
    model = DecisionTreeClassifier(
        max_depth=10,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    save_model(model, "decision_tree")
    return model

def train_random_forest(X_train, y_train, class_weight="balanced", n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs = -1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    save_model(model, "random_forest")
    return model

def train_random_forest_sm(X_train, y_train,n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs = -1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, random_state=42):
    scale_pos_weight = (y_train == 0).sum() / y_train.sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    save_model(model, "xgboost")
    return model

def train_neural_network(X_train, y_train, random_state=42):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    save_model(model, "neural_network")
    return model

