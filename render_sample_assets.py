import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'pipeline'))
sys.path.insert(0, str(REPO_ROOT / 'ui'))

from PIL import Image, ImageDraw
from server import _get_or_load_job_graph, UPLOAD_DIR, SAMPLE_DIR
from cad_schematic_renderer import render_circuit_graph

color_map = {
    'resistor': '#00f0ff',
    'capacitor': '#a855f7',
    'inductor': '#10b981',
    'voltage': '#f59e0b',
    'current': '#f59e0b',
    'transistor': '#f43f5e',
    'diode': '#ec4899',
    'gnd': '#38bdf8',
    'operational_amplifier': '#6366f1'
}

for name in ['cir1', 'cir2', 'cir3', 'cir4', 'cir5']:
    job_dir = UPLOAD_DIR / name
    job_dir.mkdir(parents=True, exist_ok=True)
    
    orig_path = SAMPLE_DIR / f'{name}.jpg'
    if orig_path.exists():
        graph = _get_or_load_job_graph(name)
        img = Image.open(orig_path).convert('RGB')
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for comp in graph.components:
            if not comp.bbox or len(comp.bbox) != 4:
                continue
            bbox = comp.bbox
            c_type = (comp.type_refined or comp.type or '').lower()
            color_hex = '#00f0ff'
            for k, v in color_map.items():
                if k in c_type:
                    color_hex = v
                    break
            rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=rgb, width=4, fill=rgb + (50,))
            
            lbl = comp.display_type or comp.type
            if comp.value:
                lbl += f': {comp.value}'
            elif comp.label:
                lbl += f': {comp.label}'
                
            tw = max(len(lbl) * 8 + 12, 50)
            th = 18
            draw.rectangle([bbox[0], max(0, bbox[1] - th - 3), bbox[0] + tw, max(0, bbox[1])], fill=rgb)
            draw.text((bbox[0] + 5, max(0, bbox[1] - th - 1)), lbl, fill=(255, 255, 255, 255))
            
            if comp.terminals:
                for term in comp.terminals:
                    px, py = getattr(term, 'x', None), getattr(term, 'y', None)
                    if px is not None and py is not None:
                        draw.ellipse([px-6, py-6, px+6, py+6], fill=(16, 185, 129, 255), outline=(255, 255, 255, 255), width=2)
                        
        annotated_file = job_dir / 'annotated.jpg'
        img.save(annotated_file, 'JPEG', quality=95)
        print(f"Generated {annotated_file}")
        
        render_circuit_graph(graph, job_dir)
        
        # Also mirror to test_previews and mcp_renders
        test_prev_dir = REPO_ROOT / 'outputs' / 'test_previews'
        test_prev_dir.mkdir(parents=True, exist_ok=True)
        render_circuit_graph(graph, test_prev_dir)
        
        mcp_dir = REPO_ROOT / 'outputs' / 'mcp_renders' / name
        mcp_dir.mkdir(parents=True, exist_ok=True)
        render_circuit_graph(graph, mcp_dir)
        
        print(f"Rendered CAD SVGs for {name}")

print("All samples successfully pre-generated!")
