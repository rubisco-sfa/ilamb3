// src/apiDirective.ts
import fs from "fs";
function optsToLabel(opts) {
  const { module, submodule, className, function: func } = opts;
  return [module, submodule, className, func].filter(Boolean).join(".");
}
function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
function labelToId(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
function sectionTitle(name) {
  return {
    type: "paragraph",
    children: [{ type: "strong", children: [{ type: "text", value: name }] }]
  };
}
function fieldToMdast(name, children) {
  return {
    type: "div",
    class: "api-field",
    children: [
      { type: "div", class: "api-field-name", children: [sectionTitle(name)] },
      { type: "div", class: "api-field-body", children }
    ]
  };
}
function classRelationToMdast(title, names, opts, documentedClasses) {
  const children = [
    { type: "strong", children: [{ type: "text", value: `${title}: ` }] }
  ];
  names.forEach((name, index) => {
    const className = { type: "inlineCode", value: name };
    if (documentedClasses.has(name)) {
      const label = optsToLabel({ ...opts, className: name, function: void 0 });
      children.push({ type: "link", url: `#${label}`, children: [className] });
    } else {
      children.push(className);
    }
    if (index < names.length - 1) children.push({ type: "text", value: ", " });
  });
  return { type: "paragraph", children };
}
function parameterToMdast(param, parse) {
  const name = param.name || param.type;
  const type = param.name && param.type ? param.type : void 0;
  const children = [
    {
      type: "strong",
      children: [{ type: "text", value: name }]
    }
  ];
  if (type) {
    children.push(
      { type: "text", value: " (" },
      { type: "emphasis", children: [{ type: "text", value: type }] },
      { type: "text", value: ")" }
    );
  }
  const description = [...parse(param.desc).children];
  const first = description.shift();
  if ((first == null ? void 0 : first.type) === "paragraph") {
    children.push({ type: "text", value: " \u2013 " }, ...first.children);
  } else if (first) {
    description.unshift(first);
  }
  return {
    type: "listItem",
    children: [
      {
        type: "paragraph",
        children
      },
      ...description
    ]
  };
}
function parameterListToMdast(name, params, parse) {
  if (params.length === 0) return [];
  return [
    fieldToMdast(name, [
      {
        type: "list",
        ordered: false,
        children: params.map((param) => parameterToMdast(param, parse))
      }
    ])
  ];
}
function functionToMdast(name, func, parse, opts, outlineDepth = opts.depth, isSubclass = false, classRelations) {
  var _a;
  const kind = (_a = func.Kind) != null ? _a : "function";
  const newOpts = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: opts.submodule,
    className: kind === "class" ? name : opts.className,
    function: kind === "class" ? void 0 : name
  };
  const signatureVariant = kind === "method" ? "api-signature-method" : kind === "class" ? isSubclass ? "api-signature-subclass" : "api-signature-base-class" : "api-signature-function";
  const signatureClass = `api-signature ${signatureVariant}`;
  const label = optsToLabel(newOpts);
  const section = kind === "method" ? [] : [{ type: "mystTarget", label }];
  if (kind !== "method") {
    const headingText = { type: "inlineCode", value: name };
    section.push({
      type: "heading",
      depth: opts.depth,
      children: outlineDepth > 4 ? [
        {
          type: "span",
          class: `api-outline-indent-${Math.min(outlineDepth - 4, 2)}`,
          children: [headingText]
        }
      ] : [headingText]
    });
  }
  const signatureId = kind === "method" ? ` id="${labelToId(label)}"` : "";
  section.push({
    type: "html",
    value: `<div class="${signatureClass}"${signatureId}><em class="api-kind">${kind}</em> <strong class="api-name"><code>${escapeHtml(label)}</code></strong><em class="api-parameters">${escapeHtml(func.Signature || "()")}</em></div>`
  });
  if (kind === "class" && classRelations) {
    const relations = [];
    if (classRelations.bases.length > 0) {
      relations.push(
        classRelationToMdast(
          "Bases",
          classRelations.bases,
          newOpts,
          classRelations.documentedClasses
        )
      );
    }
    if (classRelations.subclasses.length > 0) {
      relations.push(
        classRelationToMdast(
          "Direct subclasses",
          classRelations.subclasses,
          newOpts,
          classRelations.documentedClasses
        )
      );
    }
    section.push({ type: "div", class: "api-inheritance", children: relations });
  }
  const description = [];
  if (func.Summary)
    description.push(...parse(func.Summary.map((line) => line.trim()).join(" ")).children);
  if (typeof func["Extended Summary"] === "string") {
    description.push(...parse(func["Extended Summary"]).children);
  }
  if (description.length > 0)
    section.push({ type: "div", class: "api-description", children: description });
  if (func.Parameters) {
    section.push(...parameterListToMdast("Parameters", func.Parameters, parse));
  }
  if (func["Other Parameters"]) {
    section.push(...parameterListToMdast("Other Parameters", func["Other Parameters"], parse));
  }
  if (func.Returns) {
    section.push(...parameterListToMdast("Returns", func.Returns, parse));
  }
  if (func.Raises) {
    section.push(...parameterListToMdast("Raises", func.Raises, parse));
  }
  if (func.Warns) {
    section.push(...parameterListToMdast("Warns", func.Warns, parse));
  }
  if (typeof func.Notes === "string") {
    section.push(fieldToMdast("Notes", parse(func.Notes).children));
  }
  if (func.References) {
    const referencesAST = {
      type: "list",
      ordered: true,
      children: func.References.map((text) => ({
        type: "listItem",
        children: [
          {
            type: "paragraph",
            children: parse(text).children
          }
        ]
      }))
    };
    section.push(fieldToMdast("References", [referencesAST]));
  }
  if (func.Examples) {
    const examples = Array.isArray(func.Examples) ? func.Examples.join("\n") : func.Examples;
    section.push(fieldToMdast("Examples", parse(examples).children));
  }
  if (func["See Also"] && func["See Also"].length > 0) {
    const seeAlso = func["See Also"].flat(2).filter((val) => val.length > 0);
    const seeAlsoXrefs = seeAlso.filter((val) => typeof val !== "string").map(([val]) => val);
    const seeAlsoText = seeAlso.filter((val) => typeof val === "string").join(" ");
    if (seeAlsoXrefs.length > 0 || seeAlsoText) {
      const seeAlsoContent = [];
      if (seeAlsoXrefs.length > 0) {
        seeAlsoContent.push({
          type: "paragraph",
          children: seeAlsoXrefs.map((value, index) => {
            const xref = value.includes(".") ? value : optsToLabel({ ...opts, function: value });
            const children = [
              {
                type: "link",
                url: `#${xref}`,
                children: [
                  {
                    type: "text",
                    value
                  }
                ]
              }
            ];
            if (index < seeAlso.length - 1) {
              children.push({
                type: "text",
                value: ", "
              });
            }
            return children;
          }).flat()
        });
      }
      if (seeAlsoText) {
        seeAlsoContent.push({
          type: "paragraph",
          children: [
            {
              type: "text",
              value: seeAlsoText
            }
          ]
        });
      }
      section.push(fieldToMdast("See Also", seeAlsoContent));
    }
  }
  if (kind === "class" && func.Methods && !Array.isArray(func.Methods)) {
    const methods = [];
    Object.entries(func.Methods).forEach(([methodName, method]) => {
      methods.push(...functionToMdast(methodName, method, parse, newOpts));
    });
    section.push({ type: "div", class: "api-methods", children: methods });
  }
  return section;
}
function submoduleToMdast(name, submodule, parse, opts, includeHeading = true) {
  const newOpts = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: name
  };
  const section = [];
  if (includeHeading) {
    section.push(
      { type: "mystTarget", label: optsToLabel(newOpts) },
      {
        type: "heading",
        depth: opts.depth,
        children: [
          {
            type: "text",
            value: opts.module ? `${opts.module}.${name}` : name
          }
        ]
      }
    );
  }
  const moduleDoc = submodule.__module__;
  if ((moduleDoc == null ? void 0 : moduleDoc.Kind) === "module" && moduleDoc.Description) {
    section.push({
      type: "div",
      class: "api-module-description",
      children: parse(moduleDoc.Description).children
    });
  }
  const entries = Object.entries(submodule).filter(([, member]) => member.Kind !== "module");
  const functions = entries.filter(([, func]) => func.Kind !== "class");
  const classes = entries.filter(([, func]) => func.Kind === "class");
  const classMap = new Map(classes);
  const documentedClasses = new Set(classMap.keys());
  const classChildren = /* @__PURE__ */ new Map();
  const directSubclasses = /* @__PURE__ */ new Map();
  const classRoots = [];
  classes.forEach(([className, member]) => {
    var _a, _b, _c, _d, _e, _f;
    const directBases = (_c = (_b = member["Direct Bases"]) != null ? _b : (_a = member.Bases) == null ? void 0 : _a.slice(0, 1)) != null ? _c : [];
    const documentedBases = directBases.filter((base) => classMap.has(base));
    documentedBases.forEach((base) => {
      var _a2;
      directSubclasses.set(base, [...(_a2 = directSubclasses.get(base)) != null ? _a2 : [], className]);
    });
    const parent = (_e = documentedBases[0]) != null ? _e : (_d = member.Bases) == null ? void 0 : _d.find((base) => classMap.has(base));
    if (!parent) {
      classRoots.push([className, member]);
      return;
    }
    classChildren.set(parent, [...(_f = classChildren.get(parent)) != null ? _f : [], [className, member]]);
  });
  const sortClasses = (members) => members.sort(([left], [right]) => left.localeCompare(right));
  sortClasses(classRoots);
  classChildren.forEach(sortClasses);
  directSubclasses.forEach((subclasses) => {
    subclasses.sort((left, right) => left.localeCompare(right));
  });
  function classTreeToMdast(members, headingDepth, inheritanceDepth = headingDepth, ancestors = /* @__PURE__ */ new Set()) {
    const nodes = [];
    members.forEach(([className, member]) => {
      var _a, _b, _c, _d;
      if (ancestors.has(className)) return;
      nodes.push(
        ...functionToMdast(
          className,
          member,
          parse,
          { ...newOpts, depth: headingDepth },
          inheritanceDepth,
          inheritanceDepth > newOpts.depth,
          {
            bases: (_c = (_b = member["Direct Bases"]) != null ? _b : (_a = member.Bases) == null ? void 0 : _a.slice(0, 1)) != null ? _c : [],
            subclasses: (_d = directSubclasses.get(className)) != null ? _d : [],
            documentedClasses
          }
        )
      );
      const children = classChildren.get(className);
      if (!(children == null ? void 0 : children.length)) return;
      const nextAncestors = new Set(ancestors).add(className);
      nodes.push({
        type: "div",
        class: "api-class-children",
        children: classTreeToMdast(
          children,
          Math.min(headingDepth + 1, 4),
          inheritanceDepth + 1,
          nextAncestors
        )
      });
    });
    return nodes;
  }
  if (functions.length > 0 && classes.length > 0) {
    section.push({
      type: "div",
      class: "api-group-title",
      children: [sectionTitle("Classes")]
    });
    section.push(...classTreeToMdast(classRoots, newOpts.depth));
    section.push({
      type: "div",
      class: "api-group-title",
      children: [sectionTitle("Functions")]
    });
    functions.forEach(([memberName, member]) => {
      section.push(...functionToMdast(memberName, member, parse, newOpts));
    });
  } else {
    if (classes.length > 0) {
      section.push(...classTreeToMdast(classRoots, newOpts.depth));
    } else {
      entries.forEach(([memberName, member]) => {
        section.push(...functionToMdast(memberName, member, parse, newOpts));
      });
    }
  }
  return section;
}
function moduleToMdast(module, parse, opts) {
  const section = [];
  Object.entries(module).filter(([, submodule]) => {
    var _a;
    return ((_a = submodule.__module__) == null ? void 0 : _a.Kind) === "module";
  }).forEach(([submoduleName, submodule]) => {
    section.push(...submoduleToMdast(submoduleName, submodule, parse, opts));
  });
  return section;
}
var apiDirective = {
  name: "apidoc",
  doc: "Directive for loading docstrings (currently from fleece output)",
  arg: {
    type: String,
    doc: "File with fleece output",
    required: true
  },
  options: {
    module: {
      type: String,
      doc: "Module name for cross-reference labels."
    },
    depth: {
      type: Number,
      doc: "Starting heading depth"
    },
    layout: {
      type: String,
      doc: "API presentation layout."
    }
  },
  run(data, vfile, ctx) {
    var _a, _b, _c, _d;
    const [filename, target] = data.arg.split("#");
    const docJson = JSON.parse(fs.readFileSync(filename).toString());
    const opts = {
      depth: +((_b = (_a = data.options) == null ? void 0 : _a.depth) != null ? _b : 1),
      module: ((_c = data.options) == null ? void 0 : _c.module) ? data.options.module : void 0
    };
    if (target) {
      const [submodule, func] = target.split(".");
      if (submodule && func && ((_d = docJson[submodule]) == null ? void 0 : _d[func])) {
        return functionToMdast(func, docJson[submodule][func], ctx.parseMyst, {
          ...opts,
          submodule
        });
      }
      if (submodule && docJson[submodule]) {
        return submoduleToMdast(submodule, docJson[submodule], ctx.parseMyst, opts, false);
      }
    }
    return moduleToMdast(docJson, ctx.parseMyst, opts);
  }
};

// src/versionAdded.ts
var versionAddedDirective = {
  name: "versionadded",
  alias: ["versionchanged", "deprecated"],
  doc: "Small version added/changed/deprecated directive",
  arg: {
    type: String,
    doc: "Version the feature was added",
    required: true
  },
  run(data) {
    let verb;
    switch (data.name) {
      case "versionchanged":
        verb = "Changed";
        break;
      case "deprecated":
        verb = "Deprecated";
        break;
      default:
        verb = "Added";
    }
    return [
      {
        type: "admonition",
        kind: "note",
        children: [
          {
            type: "paragraph",
            children: [
              {
                type: "text",
                value: `${verb} in Version ${data.arg}`
              }
            ]
          }
        ]
      }
    ];
  }
};

// src/funcRole.ts
var funcRole = {
  name: "func",
  doc: "Small function cross-reference role",
  body: {
    type: String,
    doc: "Cross-reference target",
    required: true
  },
  run(data) {
    return [
      {
        type: "link",
        url: `#${data.body}`,
        children: [
          {
            type: "text",
            value: data.body
          }
        ]
      }
    ];
  }
};

// src/pyObjectRole.ts
function pyObjectRole(name, doc) {
  return {
    name,
    doc,
    body: {
      type: String,
      doc: "Python object name",
      required: true
    },
    run(data) {
      var _a;
      const target = data.body;
      const value = target.startsWith("~") ? (_a = target.slice(1).split(".").at(-1)) != null ? _a : target : target;
      return [{ type: "inlineCode", value }];
    }
  };
}
var pyClassRole = pyObjectRole("class", "Python class reference role");
var pyMethodRole = pyObjectRole("meth", "Python method reference role");
var pyModuleRole = pyObjectRole("mod", "Python module reference role");
var pyDataRole = pyObjectRole("data", "Python data reference role");

// src/index.ts
var plugin = {
  name: "Plugin to document APIs (currently using fleece output)",
  author: "mystmd developers",
  license: "MIT",
  transforms: [],
  directives: [apiDirective, versionAddedDirective],
  roles: [funcRole, pyClassRole, pyMethodRole, pyModuleRole, pyDataRole]
};
var index_default = plugin;
export {
  index_default as default
};
