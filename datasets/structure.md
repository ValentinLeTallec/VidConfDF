```
datasets
├── VideoConference
│  ├── hcvm
│  │  ├── 000
│  │  │  ├── laptop_a0117_indoor_00000_000_001.jpg
│  │  │  ...
│  │  │  └── laptop_a0397_outdoor_00169_000_001.jpg
│  │  ...
│  │  └── 085
│  │     ├── laptop_a0246_outdoor_00172_000_001.jpg
│  │     ...
│  │     └── laptop_a0246_outdoor_00178_000_001.jpg
│  │
│  ├── VideoConference-real
│  │  ├── images
│  │  │  ├── 1uedfp2lF7w@00:00:35
│  │  │  │  ├── 000
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_000_alignment.fsa
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00000_000.jpg
│  │  │  │  │  ...
│  │  │  │  │  └── 1uedfp2lF7w@00:00:35_00112_000.jpg
│  │  │  │  ├── 000_faces
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00001_000_0.png
│  │  │  │  │  ...
│  │  │  │  │  └── 1uedfp2lF7w@00:00:35_00112_000_0.png
│  │  │  │  ...
│  │  │  │  ├── 005
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00124_005.jpg
│  │  │  │  │  ...
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00284_005.jpg
│  │  │  │  │  └── 1uedfp2lF7w@00:00:35_005_alignment.fsa
│  │  │  │  ├── 005_faces
│  │  │  │  │  ├── 1uedfp2lF7w@00:00:35_00124_005_0.png
│  │  │  │  │  ...
│  │  │  │  │  └── 1uedfp2lF7w@00:00:35_00284_005_0.png
│  │  │  │  └── good
│  │  │  ...
│  │  │  └── xGq_ZHJ92fQ@00:33:10
│  │  │     ├── 000
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_000_alignment.fsa
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00000_000.jpg
│  │  │     │  ...
│  │  │     │  └── xGq_ZHJ92fQ@00:33:10_00245_000.jpg
│  │  │     ├── 000_faces
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00000_000_0.png
│  │  │     │  ...
│  │  │     │  └── xGq_ZHJ92fQ@00:33:10_00245_000_0.png
│  │  │     ...
│  │  │     ├── 002
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_002_alignment.fsa
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00246_002.jpg
│  │  │     │  ...
│  │  │     │  └── xGq_ZHJ92fQ@00:33:10_00249_002.jpg
│  │  │     ├── 002_faces
│  │  │     │  ├── xGq_ZHJ92fQ@00:33:10_00246_002_0.png
│  │  │     │  ...
│  │  │     │  └── xGq_ZHJ92fQ@00:33:10_00249_002_0.png
│  │  │     └── good
│  │  └── videos
│  │     ├── 1uedfp2lF7w@00:00:35.mp4
│  │     ...
│  │     └── xGq_ZHJ92fQ@00:33:10.mp4
│  └── VideoConference-synthesis
│     ├── images
│     │  ├── deepfakes_faceswap
│     │  │  ├── 1uedfp2lF7w@00:00:35_000⟶WhesSCRO-1M@00:15:15_002
│     │  │  │  ├── 1uedfp2lF7w@00:00:35_00000_000.jpg
│     │  │  │  ...
│     │  │  │  └── 1uedfp2lF7w@00:00:35_00112_000.jpg
│     │  │  ...
│     │  │  └── xGq_ZHJ92fQ@00:33:10_001⟶vekX_HwZtuM@01:45:20_000
│     │  │     ├── xGq_ZHJ92fQ@00:33:10_00000_001.jpg
│     │  │     ...
│     │  │     └── xGq_ZHJ92fQ@00:33:10_00245_001.jpg
│     │  └── faceshifter
│     │     ├── 1uedfp2lF7w@00:00:35_000⟶n2Ipy5bhGQQ@00:08:12_002
│     │     │  ├── 1uedfp2lF7w@00:00:35_00042_000.jpg
│     │     │  ...
│     │     │  └── 1uedfp2lF7w@00:00:35_00112_000.jpg
│     │     ...
│     │     └── xGq_ZHJ92fQ@00:33:10_002⟶H6LjpZjKURo@00:33:20_007
│     │        ├── xGq_ZHJ92fQ@00:33:10_00246_002.jpg
│     │        ...
│     │        └── xGq_ZHJ92fQ@00:33:10_00249_002.jpg
│     │
│     └── videos
│        ├── deepfakes_faceswap
│        │  ├── pt-wO73Y8to@00:09:15
│        │  │  ├── pt-wO73Y8to@00:09:15-deepfakes_faceswap-0000.mp4
│        │  │  └── pt-wO73Y8to@00:09:15-deepfakes_faceswap-0001.mp4
│        │  ...
│        │  └── vekX_HwZtuM@00:15:15
│        │     ├── vekX_HwZtuM@00:15:15-deepfakes_faceswap-0000.mp4
│        │     └── vekX_HwZtuM@00:15:15-deepfakes_faceswap-0001.mp4
│        └── faceshifter
│           ├── 1uedfp2lF7w@00:00:35
│           │  ├── 1uedfp2lF7w@00:00:35_000⟶n2Ipy5bhGQQ@00:08:12_002-faceshifter.mp4
│           │  ...
│           │  └── 1uedfp2lF7w@00:00:35_004⟶laptop_a0350_outdoor_005-faceshifter.mp4
│           ...
│           └── xGq_ZHJ92fQ@00:33:10
│              ├── xGq_ZHJ92fQ@00:33:10_000⟶laptop_a0179_indoor_005-faceshifter.mp4
│              ...
│              └── xGq_ZHJ92fQ@00:33:10_002⟶H6LjpZjKURo@00:33:20_007-faceshifter.mp4
├── VideoConference_extended
│  └── VideoConference_extended-real
│     ├── images
│     │  ├── 1uedfp2lF7w@00:00:35_extended
│     │  │  ├── 000
│     │  │  │  ├── 1uedfp2lF7w@00:00:35_extended_00000_000.jpg
│     │  │  │  ...
│     │  │  │  └── 1uedfp2lF7w@00:00:35_extended_09374_000.jpg
│     │  │  ...
│     │  │  └── 062
│     │  │     └── 1uedfp2lF7w@00:00:35_extended_00963_062.jpg
│     │  ...
│     │  └── xGq_ZHJ92fQ@00:33:10_extended
│     │     ├── 000
│     │     │  ├── xGq_ZHJ92fQ@00:33:10_extended_00000_000.jpg
│     │     │  ...
│     │     │  └── xGq_ZHJ92fQ@00:33:10_extended_05619_000.jpg
│     │     ...
│     │     ├── 008
│     │     │  ├── xGq_ZHJ92fQ@00:33:10_extended_03184_008.jpg
│     │     │  ...
│     │     │  └── xGq_ZHJ92fQ@00:33:10_extended_03187_008.jpg
│     │     └── manual
│     │        ├── 000
│     │        │  ├── xGq_ZHJ92fQ@00:33:10_extended_00000_000.jpg
│     │        │  ...
│     │        │  └── xGq_ZHJ92fQ@00:33:10_extended_05619_000.jpg
│     │        ├── 001
│     │        │  ├── xGq_ZHJ92fQ@00:33:10_extended_00000_001.jpg
│     │        │  ...
│     │        │  └── xGq_ZHJ92fQ@00:33:10_extended_05619_001.jpg
│     │        └── broken
│     │           ├── 002
│     │           │  ├── xGq_ZHJ92fQ@00:33:10_extended_00573_002.jpg
│     │           │  ...
│     │           │  └── xGq_ZHJ92fQ@00:33:10_extended_05382_002.jpg
│     │           ...
│     │           └── 008
│     │              ├── xGq_ZHJ92fQ@00:33:10_extended_03184_008.jpg
│     │              ...
│     │              └── xGq_ZHJ92fQ@00:33:10_extended_03187_008.jpg
│     └── videos
│        ├── 1uedfp2lF7w@00:00:35_extended.mp4
│        ...
│        └── xGq_ZHJ92fQ@00:33:10_extended.mp4
└── Deepfake.db
```