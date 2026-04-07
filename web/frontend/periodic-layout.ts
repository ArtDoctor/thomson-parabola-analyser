/** Standard 18-wide grid: main rows 0–6, lanthanides row 8, actinides row 9. */

export interface PeriodicCell {
  sym: string;
  z: number;
  row: number;
  col: number;
}

const SYMBOLS: readonly string[] = [
  'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
  'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
  'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
  'Ho', 'Er', 'Tm', 'Yb', 'Lu',
  'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
  'Po', 'At', 'Rn',
  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
  'Es', 'Fm', 'Md', 'No', 'Lr',
  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
  'Lv', 'Ts', 'Og',
] as const;

export const PERIODIC_CELLS: PeriodicCell[] = buildPeriodicCells();

function buildPeriodicCells(): PeriodicCell[] {
  const cells: PeriodicCell[] = [];
  const place = (z: number, row: number, col: number): void => {
    cells.push({ sym: SYMBOLS[z - 1], z, row, col });
  };

  place(1, 0, 0);
  place(2, 0, 17);
  for (let z = 3; z <= 4; z++) {
    place(z, 1, z === 3 ? 0 : 1);
  }
  for (let z = 5; z <= 10; z++) {
    place(z, 1, z + 7);
  }
  for (let z = 11; z <= 12; z++) {
    place(z, 2, z === 11 ? 0 : 1);
  }
  for (let z = 13; z <= 18; z++) {
    place(z, 2, z - 1);
  }
  for (let z = 19; z <= 20; z++) {
    place(z, 3, z === 19 ? 0 : 1);
  }
  for (let z = 21; z <= 30; z++) {
    place(z, 3, z - 19);
  }
  for (let z = 31; z <= 36; z++) {
    place(z, 3, 6 + z);
  }
  for (let z = 37; z <= 38; z++) {
    place(z, 4, z === 37 ? 0 : 1);
  }
  for (let z = 39; z <= 48; z++) {
    place(z, 4, z - 37);
  }
  for (let z = 49; z <= 54; z++) {
    place(z, 4, 6 + z);
  }
  for (let z = 55; z <= 56; z++) {
    place(z, 5, z === 55 ? 0 : 1);
  }
  place(57, 8, 3);
  for (let z = 58; z <= 71; z++) {
    place(z, 8, 3 + (z - 57));
  }
  for (let z = 72; z <= 86; z++) {
    place(z, 5, z - 69);
  }
  for (let z = 87; z <= 88; z++) {
    place(z, 6, z === 87 ? 0 : 1);
  }
  place(89, 9, 3);
  for (let z = 90; z <= 103; z++) {
    place(z, 9, 3 + (z - 89));
  }
  for (let z = 104; z <= 118; z++) {
    place(z, 6, z - 101);
  }
  return cells;
}

export const PERIODIC_TOTAL_ROWS = 10;
export const PERIODIC_TOTAL_COLS = 18;
