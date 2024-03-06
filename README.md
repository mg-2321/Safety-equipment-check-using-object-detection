# Safety-equipment-check-using-object-detection
The project aims to mitigate the issue of fatal construction accidents due to traditional manual supervision and shortage of qualified site inspectors which is augmented by developing an object detection model for smart-construction safety-inspection protocol that would identify an individual constrction worker and detect the PPEs in the form of hardhats/helmets,vests and glasses.

Algorithms such as CenterNet, Faster R-CNN and Single-Shot-Detection(SSD) have been implemented with explainable comparative analysis.

COMPARISION OF MODELS
|----------------|------------------|--------------|
|ALGORITHMS USED |AVERAGE PRESCISION|AVERAGE RECALL|
|----------------|------------------|--------------|
|  Faster R-CNN  |    57.9%         |    69.6%     |
|   (on CHVG)    |                  |              |
|----------------|------------------|--------------|
|  Faster R-CNN  |    49.3%         |    57.6%     |
|   (on PICTOR)  |                  |              |
|----------------|------------------|--------------|
|      SSD       |    53.8%         |    61.1%     |
|   (on CHVG)    |                  |              |
|----------------|------------------|--------------|
|      SSD       |    60.2%         |   62.7%      | 
|   (on PICTOR)  |                  |              |
|----------------|------------------|--------------|
|   CenterNet    |     71.1%        |   74.3%      |
|   (on CHVG)    |                  |              |
|----------------|------------------|--------------|
|   CenterNet    |     65.9%        |   69.1%      |
|   (on PICTOR)  |                  |              |
|----------------|------------------|--------------|
