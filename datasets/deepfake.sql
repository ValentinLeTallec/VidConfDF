CREATE TABLE videos( 

    videoid     INTEGER PRIMARY KEY, 
    label       TEXT    NOT NULL, 
    folder      TEXT    NOT NULL, 
    file        TEXT    NOT NULL, 
    dataset     TEXT    NOT NULL, 

    /* Fake infos (only if label = 'FAKE' */
    method      TEXT,
    targetid    INTEGER REFERENCES videos(videoid), 
    sourceid    INTEGER REFERENCES videos(videoid),   

    /* Metadata */
    width       INTEGER NOT NULL, 
    height      INTEGER NOT NULL, 
    nb_frames   INTEGER NOT NULL,
    framerate   FLOAT,
    bitrate     FLOAT,
    compression TEXT,

    CONSTRAINT unique_frame_per_video UNIQUE(dataset, folder, file)
);

CREATE INDEX idx_videos_dataset ON videos(dataset);
CREATE INDEX idx_videos_label ON videos(label);

/*
** Frame of a video
*/
CREATE TABLE frames(

    frameid  INTEGER PRIMARY KEY,
    videoid  INTEGER REFERENCES videos(videoid),
    frame_nb INTEGER NOT NULL,

    CONSTRAINT unique_frame_per_video UNIQUE(frame_nb, videoid)
);

CREATE INDEX idx_frames_videoid ON frames(videoid);


/*
** Faces per frame,
** There may be more than 1 face
*/
CREATE TABLE faces(

    faceid   INTEGER PRIMARY KEY, 
    frameid  INTEGER NOT NULL REFERENCES frame(frameid),
    folder   TEXT    NOT NULL,
    file     TEXT    NOT NULL,
    identity INTEGER NOT NULL,
    x        INTEGER NOT NULL,
    y        INTEGER NOT NULL,
    w        INTEGER NOT NULL,
    h        INTEGER NOT NULL,
    p        FLOAT   NOT NULL,


    /* Fake infos (only if isfake=1) */
    isfake      INTEGER NOT NULL DEFAULT 0,
    method      TEXT, 
    
    donor_videoid   INTEGER REFERENCES videos(videoid),
    donor_identity  INTEGER,
    
    receiver_videoid   INTEGER REFERENCES videos(videoid),
    receiver_identity  INTEGER,

    CONSTRAINT unique_frame_per_video UNIQUE(folder, file)
);
-- ALTER TABLE faces ADD COLUMN isfake INTEGER NOT NULL DEFAULT 0;

CREATE INDEX idx_faces_frameid ON faces(frameid);

/*
** Split test, train and validation 
*/
CREATE TABLE video_split(

    origin  TEXT NOT NULL,
    videoid INTEGER REFERENCES videos(videoid),  
    /*TEST, TRAIN, VALID*/  
    split   TEXT,

    CONSTRAINT unique_frame_per_video UNIQUE(origin, videoid)
);