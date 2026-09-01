document.documentElement.classList.add("js");

const toggle = document.querySelector("[data-nav-toggle]");
const navigation = document.querySelector("[data-nav]");

if (toggle && navigation) {
  const closeNavigation = () => {
    toggle.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
    document.body.classList.remove("nav-open");
  };

  toggle.addEventListener("click", () => {
    const isOpen = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!isOpen));
    navigation.classList.toggle("is-open", !isOpen);
    document.body.classList.toggle("nav-open", !isOpen);
  });

  navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeNavigation));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNavigation();
  });
}

const reveals = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window && reveals.length) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );

  reveals.forEach((element) => observer.observe(element));
} else {
  reveals.forEach((element) => element.classList.add("is-visible"));
}

const motionCanvas = document.querySelector("[data-motion-signal]");

if (motionCanvas) {
  const context = motionCanvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const colours = ["#fffdf8", "#d9f39a", "#ffb8a8"];
  let width = 0;
  let height = 0;
  let animationFrame = null;

  const activityEnvelope = (position) => {
    if (position < 0.2) return 0.08;
    if (position < 0.36) return 0.72;
    if (position < 0.7) return 0.46 + 0.22 * Math.sin(position * 37);
    if (position < 0.82) return 0.64;
    return 0.1;
  };

  const movementValue = (position, axis) => {
    const activity = activityEnvelope(position);
    const localMovement =
      Math.sin(position * (176 + axis * 19) + axis * 1.7) * 0.52 +
      Math.sin(position * (397 - axis * 23) + axis * 0.8) * 0.25 +
      Math.sin(position * 73 + axis * 2.4) * 0.14;
    const transitions =
      Math.exp(-Math.pow((position - 0.22) / 0.018, 2)) * (axis === 1 ? -0.65 : 0.55) +
      Math.exp(-Math.pow((position - 0.71) / 0.024, 2)) * (axis === 2 ? 0.7 : -0.35);

    return localMovement * activity + transitions;
  };

  const trace = (axis, end, colour, opacity, lineWidth) => {
    const baseline = height * ((axis + 0.5) / 3);
    const amplitude = height * 0.105;
    const finalX = Math.max(2, Math.floor(width * end));

    context.beginPath();
    for (let x = 0; x <= finalX; x += 2) {
      const position = x / width;
      const y = baseline - movementValue(position, axis) * amplitude;
      if (x === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.globalAlpha = opacity;
    context.strokeStyle = colour;
    context.lineWidth = lineWidth;
    context.lineJoin = "round";
    context.stroke();
  };

  const draw = (time = 0) => {
    context.clearRect(0, 0, width, height);
    const progress = reducedMotion.matches ? 1 : (time % 14000) / 14000;

    colours.forEach((colour, axis) => trace(axis, 1, colour, 0.18, 1));
    colours.forEach((colour, axis) => trace(axis, progress, colour, 0.94, 1.5));

    if (!reducedMotion.matches) {
      const cursorX = width * progress;
      context.beginPath();
      context.moveTo(cursorX, 0);
      context.lineTo(cursorX, height);
      context.globalAlpha = 0.55;
      context.strokeStyle = "#fffdf8";
      context.lineWidth = 1;
      context.stroke();
    }

    context.globalAlpha = 1;
  };

  const animate = (time) => {
    draw(time);
    animationFrame = window.requestAnimationFrame(animate);
  };

  const resizeCanvas = () => {
    const bounds = motionCanvas.getBoundingClientRect();
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);
    motionCanvas.width = Math.round(width * scale);
    motionCanvas.height = Math.round(height * scale);
    context.setTransform(scale, 0, 0, scale, 0, 0);
    draw();
  };

  const updateAnimation = () => {
    if (animationFrame) window.cancelAnimationFrame(animationFrame);
    animationFrame = null;
    if (reducedMotion.matches) draw();
    else animationFrame = window.requestAnimationFrame(animate);
  };

  resizeCanvas();
  updateAnimation();
  window.addEventListener("resize", resizeCanvas, { passive: true });
  reducedMotion.addEventListener("change", updateAnimation);
}
