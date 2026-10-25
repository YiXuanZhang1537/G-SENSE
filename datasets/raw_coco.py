import os

from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # self.id_to_patient = {image["id"]: image["patient_id"] for image in self.coco.dataset["images"]}


    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        patient_id=index
        results = [d for d in self.coco.dataset["images"]  if d['view_id'] == index]
        CC_path="abc"
        ML_path="abc"
        if len(results) == 2:
            for result in results:
                if result['view'] == 'CC':
                    CC_id=result['id']
                    CC_path=os.path.join(self.root, result['file_name'])
                elif result['view'] == 'MLO':
                    ML_id=result['id']
                    ML_path=os.path.join(self.root, result['file_name'])
                else:
                    print("unknown view!!!!")
        else:
            print("error!!!!!!")
            print(results)
            print(patient_id)

        # print(CC_path, ML_path)
        CC_img=Image.open(CC_path).convert("RGB")
        ML_img=Image.open(ML_path).convert("RGB")
        # image = self._load_image(id)
        CC_target = self._load_target(CC_id)
        ML_target = self._load_target(ML_id)

        if self.transforms is not None:
            CC_img,ML_img,CC_target,ML_target = self.transforms(CC_img, ML_img, CC_target, ML_target)

        return CC_img,ML_img,CC_id,ML_id,CC_target,ML_target,patient_id

    def __len__(self) -> int:
        length=int(len(self.ids)/2)
        return length