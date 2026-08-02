import { useEffect, useRef } from "react";

interface Node {
  x: number; y: number; vx: number; vy: number;
}
interface Pulse {
  from: number; to: number; progress: number; speed: number;
}

export default function NetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = (canvas.width = canvas.offsetWidth * 2);
    let height = (canvas.height = canvas.offsetHeight * 2);
    ctx.scale(2, 2);

    const NODE_COUNT = 26;
    const nodes: Node[] = Array.from({ length: NODE_COUNT }, () => ({
      x: Math.random() * (width / 2),
      y: Math.random() * (height / 2),
      vx: (Math.random() - 0.5) * 0.15,
      vy: (Math.random() - 0.5) * 0.15,
    }));

    const edges: [number, number][] = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      const connections = 1 + Math.floor(Math.random() * 2);
      for (let c = 0; c < connections; c++) {
        const j = Math.floor(Math.random() * NODE_COUNT);
        if (j !== i) edges.push([i, j]);
      }
    }

    const pulses: Pulse[] = [];
    const spawnPulse = () => {
      const [from, to] = edges[Math.floor(Math.random() * edges.length)];
      pulses.push({ from, to, progress: 0, speed: 0.006 + Math.random() * 0.008 });
    };
    const spawnInterval = setInterval(spawnPulse, 350);

    let raf: number;
    const render = () => {
      ctx.clearRect(0, 0, width / 2, height / 2);

      nodes.forEach((n) => {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > width / 2) n.vx *= -1;
        if (n.y < 0 || n.y > height / 2) n.vy *= -1;
      });

      ctx.strokeStyle = "rgba(34, 211, 238, 0.08)";
      ctx.lineWidth = 1;
      edges.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo(nodes[a].x, nodes[a].y);
        ctx.lineTo(nodes[b].x, nodes[b].y);
        ctx.stroke();
      });

      for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i];
        p.progress += p.speed;
        if (p.progress >= 1) { pulses.splice(i, 1); continue; }
        const a = nodes[p.from], b = nodes[p.to];
        const x = a.x + (b.x - a.x) * p.progress;
        const y = a.y + (b.y - a.y) * p.progress;
        const grad = ctx.createRadialGradient(x, y, 0, x, y, 6);
        grad.addColorStop(0, "rgba(34, 211, 238, 0.9)");
        grad.addColorStop(1, "rgba(34, 211, 238, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      nodes.forEach((n) => {
        ctx.fillStyle = "rgba(124, 138, 165, 0.5)";
        ctx.beginPath();
        ctx.arc(n.x, n.y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      });

      raf = requestAnimationFrame(render);
    };
    render();

    return () => { cancelAnimationFrame(raf); clearInterval(spawnInterval); };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full opacity-60 pointer-events-none"
    />
  );
}