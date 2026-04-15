---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Setting Global `ilamb3` Options

```{warning} Unfinished
This feature is not in its final form. We will provide some details here, but expect this to change in the future.
```

While much of the `ilamb3` benchmark study may be controlled in the yaml benchmark [configure](configure_yaml) file, it is often more useful and less verbose to set global options. `ilamb3` employes a system where a global dictionary may be accessed and changed. To see what options are set, from within a python interpreter run the following commands:

```{code-cell} python
import ilamb3
print(ilamb3.conf)
```

This output is the configuration dictionary displayed in yaml syntax. Currently, there is limited control of these options from the `ilamb run` command. For example, the `--cache|--no-cache` option in the `run` command will set/unset the `use_cached_results` option. Eventually we will expand the system so that more can be controlled.
