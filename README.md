# Skull2Face: geodesic grid as intermediary
This code is a python implementation of the "Craniofacial reconstruction based on heat flow geodesic grid regression (HF-GGR) model".

Given dense correspondence 3D geometry data of skull, this implemented module facilitate the prediction of 3D geometry of human face. The intermediary connection is established through geodesic grid points radially distributed on the skin.   

It's worth noting that there exists subtle differences between the published paper and the implementation. The algorithm for calculating geodesic distance, originally based on heat flow, has been substituted with the geometric distance measured along the mesh edges.



## References
```
@article{jia2021craniofacial,
  title={Craniofacial reconstruction based on heat flow geodesic grid regression (HF-GGR) model},
  author={Jia, Bin and Zhao, Junli and Xin, Shiqing and Duan, Fuqing and Pan, Zhenkuan and Wu, Zhongke and Li, Jinhua and Zhou, Mingquan},
  journal={Computers \& Graphics},
  volume={97},
  pages={258--267},
  year={2021},
  publisher={Elsevier}
}
```

