import fs from 'node:fs';
import type {
  DirectiveData,
  DirectiveSpec,
  GenericNode,
  GenericParent,
  DirectiveContext,
} from 'myst-common';
import type { Func, Module, Options, Parameter, Parser, Submodule } from './types.js';
import type { VFile } from 'vfile';

export function optsToLabel(opts: Options) {
  const { module, submodule, className, function: func } = opts;
  return [module, submodule, className, func].filter(Boolean).join('.');
}

function escapeHtml(value: string) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function labelToId(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function sectionTitle(name: string): GenericNode {
  return {
    type: 'paragraph',
    children: [{ type: 'strong', children: [{ type: 'text', value: name }] }],
  };
}

function fieldToMdast(name: string, children: GenericNode[]): GenericNode {
  return {
    type: 'div',
    class: 'api-field',
    children: [
      { type: 'div', class: 'api-field-name', children: [sectionTitle(name)] },
      { type: 'div', class: 'api-field-body', children },
    ],
  };
}

export function parameterToMdast(param: Parameter, parse: Parser): GenericNode {
  const name = param.name || param.type; // Sometimes name is "" and type should be used for the name
  const type = param.name && param.type ? param.type : undefined;
  const children: GenericNode[] = [
    {
      type: 'strong',
      children: [{ type: 'text', value: name }],
    },
  ];
  if (type) {
    children.push(
      { type: 'text', value: ' (' },
      { type: 'emphasis', children: [{ type: 'text', value: type }] },
      { type: 'text', value: ')' },
    );
  }

  const description = [...parse(param.desc).children];
  const first = description.shift();
  if (first?.type === 'paragraph') {
    children.push({ type: 'text', value: ' – ' }, ...(first as GenericParent).children);
  } else if (first) {
    description.unshift(first);
  }

  return {
    type: 'listItem',
    children: [
      {
        type: 'paragraph',
        children,
      },
      ...description,
    ],
  };
}

export function parameterListToMdast(
  name: string,
  params: Parameter[],
  parse: Parser,
): GenericNode[] {
  if (params.length === 0) return [];
  return [
    fieldToMdast(name, [
      {
        type: 'list',
        ordered: false,
        children: params.map((param) => parameterToMdast(param, parse)),
      },
    ]),
  ];
}

export function functionToMdast(
  name: string,
  func: Func,
  parse: Parser,
  opts: Options,
): GenericNode[] {
  const kind = func.Kind ?? 'function';
  const newOpts: Options = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: opts.submodule,
    className: kind === 'class' ? name : opts.className,
    function: kind === 'class' ? undefined : name,
  };
  const signatureClass = kind === 'method' ? 'api-signature api-signature-method' : 'api-signature';
  const label = optsToLabel(newOpts);
  const section: GenericNode[] = kind === 'method' ? [] : [{ type: 'mystTarget', label }];
  if (kind !== 'method') {
    section.push({
      type: 'heading',
      depth: opts.depth,
      children: [{ type: 'inlineCode', value: name }],
    });
  }
  const signatureId = kind === 'method' ? ` id="${labelToId(label)}"` : '';
  section.push({
    type: 'html',
    value: `<div class="${signatureClass}"${signatureId}><em class="api-kind">${kind}</em> <strong class="api-name"><code>${escapeHtml(label)}</code></strong><em class="api-parameters">${escapeHtml(func.Signature || '()')}</em></div>`,
  });
  const description: GenericNode[] = [];
  if (func.Summary)
    description.push(...parse(func.Summary.map((line) => line.trim()).join(' ')).children);
  if (typeof func['Extended Summary'] === 'string') {
    description.push(...parse(func['Extended Summary']).children);
  }
  if (description.length > 0)
    section.push({ type: 'div', class: 'api-description', children: description });
  if (func.Parameters) {
    section.push(...parameterListToMdast('Parameters', func.Parameters, parse));
  }
  if (func['Other Parameters']) {
    section.push(...parameterListToMdast('Other Parameters', func['Other Parameters'], parse));
  }
  if (func.Returns) {
    section.push(...parameterListToMdast('Returns', func.Returns, parse));
  }
  if (func.Raises) {
    section.push(...parameterListToMdast('Raises', func.Raises, parse));
  }
  if (func.Warns) {
    section.push(...parameterListToMdast('Warns', func.Warns, parse));
  }
  if (typeof func.Notes === 'string') {
    section.push(fieldToMdast('Notes', parse(func.Notes).children));
  }
  if (func.References) {
    const referencesAST = {
      type: 'list',
      ordered: true,
      children: func.References.map((text) => ({
        type: 'listItem',
        children: [
          {
            type: 'paragraph',
            children: parse(text).children,
          },
        ],
      })),
    };
    section.push(fieldToMdast('References', [referencesAST]));
  }
  if (func.Examples) {
    section.push(
      fieldToMdast('Examples', [
        {
          type: 'code',
          lang: 'python',
          value: func.Examples.join('\n'),
        },
      ]),
    );
  }
  if (func['See Also'] && func['See Also'].length > 0) {
    const seeAlso = func['See Also'].flat(2).filter((val) => val.length > 0);
    const seeAlsoXrefs = seeAlso
      .filter((val): val is [string, null] => typeof val !== 'string')
      .map(([val]) => val);
    const seeAlsoText = seeAlso.filter((val) => typeof val === 'string').join(' ');
    if (seeAlsoXrefs.length > 0 || seeAlsoText) {
      const seeAlsoContent: GenericNode[] = [];
      if (seeAlsoXrefs.length > 0) {
        seeAlsoContent.push({
          type: 'paragraph',
          children: seeAlsoXrefs
            .map((value, index) => {
              const xref = value.includes('.') ? value : optsToLabel({ ...opts, function: value });
              const children: GenericNode[] = [
                {
                  type: 'link',
                  url: `#${xref}`,
                  children: [
                    {
                      type: 'text',
                      value,
                    },
                  ],
                },
              ];
              if (index < seeAlso.length - 1) {
                children.push({
                  type: 'text',
                  value: ', ',
                });
              }
              return children;
            })
            .flat(),
        });
      }
      if (seeAlsoText) {
        seeAlsoContent.push({
          type: 'paragraph',
          children: [
            {
              type: 'text',
              value: seeAlsoText,
            },
          ],
        });
      }
      section.push(fieldToMdast('See Also', seeAlsoContent));
    }
  }
  if (kind === 'class' && func.Methods && !Array.isArray(func.Methods)) {
    const methods: GenericNode[] = [];
    Object.entries(func.Methods).forEach(([methodName, method]) => {
      methods.push(...functionToMdast(methodName, method, parse, newOpts));
    });
    section.push({ type: 'div', class: 'api-methods', children: methods });
  }
  return section;
}

export function submoduleToMdast(
  name: string,
  submodule: Submodule,
  parse: Parser,
  opts: Options,
  includeHeading = true,
): GenericNode[] {
  const newOpts = {
    depth: opts.depth + 1,
    module: opts.module,
    submodule: name,
  };
  const section: GenericNode[] = [];
  if (includeHeading) {
    section.push(
      { type: 'mystTarget', label: optsToLabel(newOpts) },
      {
        type: 'heading',
        depth: opts.depth,
        children: [
          {
            type: 'text',
            value: opts.module ? `${opts.module}.${name}` : name,
          },
        ],
      },
    );
  }
  const moduleDoc = submodule.__module__;
  if (moduleDoc?.Kind === 'module' && moduleDoc.Description) {
    section.push({
      type: 'div',
      class: 'api-module-description',
      children: parse(moduleDoc.Description).children,
    });
  }
  const entries = Object.entries(submodule).filter(([, member]) => member.Kind !== 'module');
  const functions = entries.filter(([, func]) => func.Kind !== 'class');
  const classes = entries.filter(([, func]) => func.Kind === 'class');
  const classMap = new Map(classes);
  const classChildren = new Map<string, [string, Func][]>();
  const classRoots: [string, Func][] = [];
  classes.forEach(([className, member]) => {
    const parent = member.Bases?.find((base) => classMap.has(base));
    if (!parent) {
      classRoots.push([className, member]);
      return;
    }
    classChildren.set(parent, [...(classChildren.get(parent) ?? []), [className, member]]);
  });
  const sortClasses = (members: [string, Func][]) =>
    members.sort(([left], [right]) => left.localeCompare(right));
  sortClasses(classRoots);
  classChildren.forEach(sortClasses);

  function classTreeToMdast(
    members: [string, Func][],
    depth: number,
    ancestors = new Set<string>(),
  ): GenericNode[] {
    const nodes: GenericNode[] = [];
    members.forEach(([className, member]) => {
      if (ancestors.has(className)) return;
      nodes.push(...functionToMdast(className, member, parse, { ...newOpts, depth }));
      const children = classChildren.get(className);
      if (!children?.length) return;
      const nextAncestors = new Set(ancestors).add(className);
      nodes.push({
        type: 'div',
        class: 'api-class-children',
        children: classTreeToMdast(children, Math.min(depth + 1, 6), nextAncestors),
      });
    });
    return nodes;
  }
  if (functions.length > 0 && classes.length > 0) {
    section.push({
      type: 'div',
      class: 'api-group-title',
      children: [sectionTitle('Classes')],
    });
    section.push(...classTreeToMdast(classRoots, newOpts.depth));
    section.push({
      type: 'div',
      class: 'api-group-title',
      children: [sectionTitle('Functions')],
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

export function moduleToMdast(module: Module, parse: Parser, opts: Options): GenericNode[] {
  const section: GenericNode[] = [];
  Object.entries(module)
    .filter(([, submodule]) => submodule.__module__?.Kind === 'module')
    .forEach(([submoduleName, submodule]) => {
      section.push(...submoduleToMdast(submoduleName, submodule, parse, opts));
    });
  return section;
}

export const apiDirective: DirectiveSpec = {
  name: 'apidoc',
  doc: 'Directive for loading docstrings (currently from fleece output)',
  arg: {
    type: String,
    doc: 'File with fleece output',
    required: true,
  },
  options: {
    module: {
      type: String,
      doc: 'Module name for cross-reference labels.',
    },
    depth: {
      type: Number,
      doc: 'Starting heading depth',
    },
    layout: {
      type: String,
      doc: 'API presentation layout.',
    },
  },
  run(data: DirectiveData, vfile: VFile, ctx: DirectiveContext) {
    const [filename, target] = (data.arg as string).split('#');
    const docJson = JSON.parse(fs.readFileSync(filename).toString());
    const opts = {
      depth: +(data.options?.depth ?? 1),
      module: data.options?.module ? (data.options.module as string) : undefined,
    };
    if (target) {
      const [submodule, func] = target.split('.');
      if (submodule && func && docJson[submodule]?.[func]) {
        return functionToMdast(func, docJson[submodule][func], ctx.parseMyst, {
          ...opts,
          submodule,
        });
      }
      if (submodule && docJson[submodule]) {
        return submoduleToMdast(submodule, docJson[submodule], ctx.parseMyst, opts, false);
      }
    }
    return moduleToMdast(docJson, ctx.parseMyst, opts);
  },
};
