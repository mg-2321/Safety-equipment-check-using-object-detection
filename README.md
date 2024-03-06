# Safety-equipment-check-using-object-detection
The project aims to mitigate the issue of fatal construction accidents due to traditional manual supervision and shortage of qualified site inspectors which is augmented by developing an object detection model for smart-construction safety-inspection protocol that would identify an individual constrction worker and detect the PPEs in the form of hardhats/helmets,vests and glasses.

Algorithms such as CenterNet, Faster R-CNN and Single-Shot-Detection(SSD) have been implemented with explainable comparative analysis.

COMPARISION OF MODELS



|     ALGORITHMS USED     |AVERAGE PRESCISION|AVERAGE RECALL|
|-------------------------|------------------|--------------|
|  Faster R-CNN(on CHVG)  |    57.9%         |    69.6%     |
|-------------------------|------------------|--------------|
|  Faster R-CNN(on PICTOR)|    49.3%         |    57.6%     |
|-------------------------|------------------|--------------|
|      SSD(on CHGV)       |    53.8%         |    61.1%     |
|-------------------------|------------------|--------------|
|      SSD(on PICTOR)     |    60.2%         |   62.7%      | 
|-------------------------|------------------|--------------|
|   CenterNet(on CHGV)    |     71.1%        |   74.3%      |
|-------------------------|------------------|--------------|
|   CenterNet(on PICTOR)  |     65.9%        |   69.1%      |
|-------------------------|------------------|--------------|
