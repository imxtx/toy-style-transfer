# Toy Style Transfer

Requirements: PyTorch >= 1.12.1

Usage:

```bash
python main.py -c images/mbg.png -s images/style.jpg -o images/mbg_cool.png -d cuda:0
```

Print usage message:

```bash
python main.py -h
```

## Result

<div>
    <img src="./images/mbg.png" height=250></img>
    <img src="./images/style.jpg" width=250 height=250></img>
    <img src="./images/mbg_cool.png" height=250></img>
</div>

## References

- <https://pytorch.org/tutorials/advanced/neural_style_tutorial.html>
