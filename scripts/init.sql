CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE inference (
    inference_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    query_id VARCHAR(100) NOT NULL,
    batch BOOLEAN NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(10) NOT NULL,
    designation TEXT NOT NULL,
    description TEXT,
    predicted_category VARCHAR(50) NOT NULL,
    labeled_category VARCHAR(50),
    confidence FLOAT NOT NULL,
    queried_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    FOREIGN KEY (user_id) REFERENCES users(user_id)
);


CREATE INDEX ix_users_username ON users (username);


INSERT INTO users (
    username,
    password_hash,
    role
)
VALUES
    ('charlie@golden-ticket.com', '$2b$12$SEW5g7rqYnl7KAznnphOKur1JUCBEBkEKyplyi8eNKASskFTyHsUq', 'user'),
    ('alice@wonderland.com', '$2b$12$zBfXsQSsdGkgSVbv1vlVd..GYknETgllr3tJwSr7CQT03JH8pi/zW', 'user'),
    ('rbanat', '$2b$12$H443cg6hZgttgptlF9UqfOxRZBiy0AdMfCry5o8kUe7Qxxnwrrgte', 'admin'),
    ('walter@white.bb', '$2b$12$83YZKlHc3bHUTe3tZyYcFOij3XaWTf6JGOqmmVbrhE.qdNIUxucM.', 'user'),
    ('mallory@jewelery.gold', '$2b$12$QK9QVgAhAuW3hCbO4ih/Rejoo3Cb7mDev4mhSJ6PjtQ6o2lSC1Mb6', 'user'),
    ('david@alabama.io', '$2b$12$up1G7i2B1QhMYycMseWMPu8qoSiVmtrSxkZrNc5GXUn2czXjuYd0S', 'user'),
    ('bkhan', '$2b$12$4la4fjepQKlzIn3AGjLK5eFQ3pY8n6ZN0PIRG.xouWqg9Uazz9s2G', 'admin'),
    ('trent@trend.tt', '$2b$12$l7Tz7Qfo4n9TCHcpPtb6weSJloNnVmjJ.dBN3uKObPDYHGEmVTo42', 'user'),
    ('rmazoyer', '$2b$12$vv5J7ngaGcqeui3PqYGWo.PECwxIQHao/2magQM1/fsl4fH.0b1/O', 'admin'),
    ('franck@osc.dub', '$2b$12$v6QJw0dJ9TZtqDlwyQEK9.EIbhQZUlS/X6ttYIKXWQXr2a4.1XC6e', 'user'),
    ('strincal', '$2b$12$EhZlmg2rHefgxI2MKH.mbOec7joigRkHp7JtimZYzIa1L7WtZxGJC', 'admin'),
    ('victor@wemby.vw', '$2b$12$1aCBhe5s7sxQKHbX1kf18OnMk0mqV.LeR3QSmaouG3wjQnkYBRljG', 'user'),
    ('eve@adam.god', '$2b$12$3G5IcVpZ52ZR.3O5aYdYQuBEA6kcK9p7Cw7VBElTJNGllrZiuFAPm', 'user'),
    ('bob@marley.fr', '$2b$12$szqBL5qqV9xFcqhWhEcZnOaHw8xTAMQsO8G7owHNhERCCF86b.E42', 'user')
;
