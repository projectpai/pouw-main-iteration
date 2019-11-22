## Filler tasks mini-POC

The file `template.jsonnet` is the template for generating a client task with random ML parameters (with the seed based
on a hash value).

We use `jsonnet` placeholders (`std.extVar`) to inject external values. The external values are produced in
 `generator.py` file.