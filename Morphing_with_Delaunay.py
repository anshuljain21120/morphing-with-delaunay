import cv2
import argparse
import numpy as np

def show_image(name,image):
    '''This also destroys all currently active windows.'''
    cv2.imshow(name,image)
    cv2.imwrite(name+'.jpg',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_selection(event, x, y, flags, params):
    '''Works with only mouse-left-click'''
    img, r , marker_list, name, clr = params
    if event == cv2.EVENT_LBUTTONDOWN:
        marker_list.append([y,x])
        if len(img.shape) == 3:
            img[y-r:y+r, x-r:x+r] = list(clr)
        else:
            img[y-r:y+r, x-r:x+r] = np.iinfo(img.dtype).min
            img[y-r//2:y+r//2, x-r//2:x+r//2] = np.iinfo(img.dtype).max
        cv2.imshow(name, img)
        cv2.imwrite("Control_point_"+name+'.jpg', img)

def add_corner_points(pts, dim):
    '''Parameters are passed by reference here.'''
    x, y = dim[0], dim[1]
    pts += [[0, 0]]
    pts += [[x-1, 0]]
    pts += [[x-1, y-1]]
    pts += [[0, y-1]]

def delaunay(c = np.asarray([0, 0]), r = 9999):
    '''Initialize coordinates, centre & triangles lists.'''
    cps = [c + r * np.array((-1, -1)),
            c + r * np.array((1, -1)),
            c + r * np.array((1, 1)),
            c + r * np.array((-1, 1))]
    tri_data, circumcircles = {}, {}
    
    """Used the CCW representation of tri_data"""
    tri_data[(0, 1, 3)] = [(2, 3, 1), None, None]
    tri_data[(2, 3, 1)] = [(0, 1, 3), None, None]

    for triangle in tri_data:
        circumcircles[triangle] = circumcenter(triangle, cps)
    return cps,tri_data,circumcircles

def circumcenter(triangles,crds):
    """Calculate circumcenter using Barycentric coordinates of triangle"""

    points = np.asarray([crds[vertex] for vertex in triangles])

    """Used Cramer's Rule"""
    
    upper = np.hstack([2*np.dot(points, points.T), np.asarray([[1]*3]).T])
    A_mat = np.vstack((upper, np.asarray(3*[1] + [0])))
    b_mat = np.hstack((np.sum(points * points, axis=1), [1]))
    bary_crds = np.linalg.solve(A_mat, b_mat)[:-1]
    
    c = np.dot(bary_crds, points)
    r = np.sum(np.square(points[0] - c))
    
    return (c, r)

def is_in_circle(point,c,triangles):
    d_vec = c[triangles][0]-point
    return np.sum(np.square(d_vec)) <= c[triangles][1]

def triangulation(points,crds,triangles,circles):
    """
    Add the points to the DT and use Bowyer-Watson.
    crds = Control points list, op_tri = Opposite Triangle
    edge1, edge2 = edges, returns np array of tuple of idxs of 
    control points which formed triangle
    """
    for p in points:
        crds.append(p)
        
        '''Find conflicting triangles'''
        wrong_tri = []
        for triangle in triangles:
            if is_in_circle(p,circles,triangle):
                wrong_tri.append(triangle)
                
        '''Track polygon of infected area by checking
           all edges of conflicting triangles.'''
        polygon = []
        T = wrong_tri[0]
        e = 0
        while True:
            op_tri = triangles[T][e]
            if op_tri not in wrong_tri:
                polygon.append((T[(e+1) % 3], T[(e-1) % 3], op_tri))
                e = (e + 1) % 3
                if polygon[0][0] == polygon[-1][1]:
                    break
            else:
                e = (triangles[op_tri].index(T) + 1) % 3
                T = op_tri

        '''All conflicting triangles should be removed.'''
        for triangle in wrong_tri:
            del triangles[triangle]
            del circles[triangle]

        '''Build new ones in infected area.'''
        nT = []
        for (edge1, edge2, op_tri) in polygon:
            triangle = (len(crds)-1, edge1, edge2)
            circles[triangle] = circumcenter(triangle,crds)
            triangles[triangle] = [op_tri, None, None]
            if op_tri:
                """If op_tri exists update its neighbours for new Tris"""
                i=0
                for adj_triangle in triangles[op_tri]:
                    if adj_triangle:
                        if edge2 in adj_triangle and edge1 in adj_triangle: 
                            """Hence adj_triangle is old triangle that needs to be replaced"""
                            triangles[op_tri][i] = triangle
                    i+=1
            nT.append(triangle)

        """Update neighbour info of new Tris for new Tris"""
        for i, j in enumerate(nT):
            triangles[j][1] = nT[(i+1) % len(nT)]
            triangles[j][2] = nT[(i-1) % len(nT)]
    return [(v0-4, v1-4, v2-4) for (v0, v1, v2) in triangles if v0 > 3 and v1 > 3 and v2 > 3]

def draw_triangles(img, ctrl_pts, clr=(0,255,0), lw=1, triangles={}):
    '''A new Image and will be returned along with Triangles.
        Triangles will be calculated only if not given, else same will be returned'''
    iout = np.copy(img)
    clr_code = list(clr)
       
    '''Do Delaunay Triangulation if not provided'''
    if len(triangles) == 0:
        crds, triangles, circles = delaunay()
        triangles = triangulation(np.flip(ctrl_pts, 1),crds,triangles,circles)
    
    '''Make coordinates compatible to cv2 library'''
    ctrl_pts = np.flip(ctrl_pts, 1)
    for p1, p2, p3 in triangles:
        cv2.line(iout, tuple(ctrl_pts[p1]), tuple(ctrl_pts[p2]), clr_code, lw, cv2.LINE_AA)
        cv2.line(iout, tuple(ctrl_pts[p2]), tuple(ctrl_pts[p3]), clr_code, lw, cv2.LINE_AA)
        cv2.line(iout, tuple(ctrl_pts[p3]), tuple(ctrl_pts[p1]), clr_code, lw, cv2.LINE_AA)
    
    return iout, triangles

def area(x, y, z):
    '''Using ShoeLace Formula'''
    return abs((x[0] * (y[1] - z[1]) + y[0] * (z[1] - x[1])+ z[0] * (x[1] - y[1])) / 2.0)

def is_inside_triangle(p, a, b, c):
    '''point lies inside a triangle or not'''
    return (area(a, b, c) == (area(p, b, c) + area(a, p, c) + area(a, b, p)))

def get_triangle(P, cps, tri_list):
    for i, j, k in tri_list:
        if is_inside_triangle(P, cps[i], cps[j], cps[k]):
            return i, j, k

def confine(x, dt=np.uint8):
    '''To handle overflow and underflow of pixel values.'''
    return np.floor(max(min(x, np.iinfo(dt).max), np.iinfo(dt).min))

def framesCalculation(N, imgi, imgf, cpsi, cpsf, tri_list):
    '''Takes Considerable time. Outputs only after calculating all frames.'''
    
    frames = np.full(tuple([N]+list(imgi.shape)), np.zeros_like(imgi))
    frames[0], frames[N-1] = imgi, imgf
    delta = np.divide((cpsf-cpsi), N-1)
    coloured = (len(imgi.shape) == 3)

    for k in range(1, N-1):

        '''P[K] = P[0] + K * ΔL, where P[K] is control points vector for Kth frame.'''
        cpsk = np.add(cpsi, np.floor(np.multiply(k, delta)))

        for v0, v1, v2 in tri_list:
            '''Calculate dimensions of bounding box of triangle T'''
            mnwidth = (min(cpsk[v0][0], cpsk[v1][0], cpsk[v2][0])).astype(np.int64)
            mxwidth = (max(cpsk[v0][0], cpsk[v1][0], cpsk[v2][0])).astype(np.int64)

            mnheight = (min(cpsk[v0][1], cpsk[v1][1], cpsk[v2][1])).astype(np.int64)
            mxheight = (max(cpsk[v0][1], cpsk[v1][1], cpsk[v2][1])).astype(np.int64)

            '''Calculate Affine basis for Kth, initial & final frames'''
            e1, e2 = np.subtract(cpsk[v1], cpsk[v0]), np.subtract(cpsk[v2], cpsk[v0])
            ei1, ei2 = np.subtract(cpsi[v1], cpsi[v0]), np.subtract(cpsi[v2], cpsi[v0])
            ef1, ef2 = np.subtract(cpsf[v1], cpsf[v0]), np.subtract(cpsf[v2], cpsf[v0])

            for x in range(mnwidth, mxwidth+1):
                for y in range(mnheight, mxheight+1):

                    if not is_inside_triangle([x, y], cpsk[v0], cpsk[v1], cpsk[v2]):
                        continue

                    p = np.asarray([x, y], dtype=np.int64)

                    '''P[K] - P0[K] = α*e1[K] + β*e2[K], where α,β are Affine Coordinate Points (acps)'''
                    acps = np.linalg.solve(np.transpose([e1, e2]), np.subtract(p, cpsk[v0]))

                    pi = np.floor(np.add(np.dot(np.transpose([ei1, ei2]), acps), cpsi[v0])).astype(np.int64)
                    pf = np.floor(np.add(np.dot(np.transpose([ef1, ef2]), acps), cpsf[v0])).astype(np.int64)

                    '''Assigning colour to pixel p'''
                    w = k / (N-1)
                    frames[k][p[0],p[1]] = np.add(np.multiply(1-w, imgi[pi[0],pi[1]]), 
                                                  np.multiply(w, imgf[pf[0],pf[1]]),
                                                  dtype=imgi.dtype, casting='unsafe')

        progress = round(100 * k / (N-2)) 
        print("\r {0:}/{1:} : [{2:100s}]".format(k+2, N, '='*progress), end='')
    print()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Morphing of two Images using Delaunay Triangulation \n-by 17ucs029, 17ucs028")
    parser.add_argument("ipath",help="Path of initial image")
    parser.add_argument("fpath",help="Path of final image")
    parser.add_argument("N",help="No. of frames to generate, must be > 1",type=int)
    parser.add_argument("-outf","--outfolder",action="store",default="Morphing_Output",
                        help="Output folder path. If not given frames will be dumped to Morphing_Output/ in current directory.")
    parser.add_argument("-m","--monochrome",action="store_true",help="Ignore colour of images")
    args = parser.parse_args()

    # Load Images
    if(args.monochrome):
        img1, img2 = cv2.imread(args.ipath, 0), cv2.imread(args.fpath, 0)
    else:
        img1, img2 = cv2.imread(args.ipath), cv2.imread(args.fpath)
    assert (img1.shape == img2.shape), "Images have different dimensions!"

    # Mark Control Points
    init_ctrl_pts, final_ctrl_pts = [], []
    cv2.namedWindow('Initial Image')
    cv2.setMouseCallback('Initial Image', feature_selection, (np.copy(img1), 3, init_ctrl_pts, 'Initial Image',(0,0,255)))
    cv2.imshow('Initial Image',img1)

    cv2.namedWindow('Final Image')
    cv2.setMouseCallback('Final Image', feature_selection, (np.copy(img2), 3, final_ctrl_pts, 'Final Image', (255,0,0)))
    show_image('Final Image',img2)
    assert (len(init_ctrl_pts) == len(final_ctrl_pts)), "Images have different number of control points!"

    # Add Corner points of images
    add_corner_points(init_ctrl_pts, img1.shape)
    add_corner_points(final_ctrl_pts, img2.shape)
    init_ctrl_pts, final_ctrl_pts = np.asarray(init_ctrl_pts), np.asarray(final_ctrl_pts)

    # Draw Triangles using Delaunay Triangulation
    img, triangles = draw_triangles(img1,init_ctrl_pts.copy(),clr=(0,0,255))
    cv2.imshow("Initial Image with Triangles", img)
    cv2.imwrite("Initial Image with Triangles.jpg",img)
    img, triangles = draw_triangles(img2,final_ctrl_pts.copy(),clr=(255,0,0),triangles=triangles)
    show_image("Final Image with Triangles", img)

    # Input number of frames to generate
    N = args.N
    assert (N > 1), "N should be greater than 1"

    # Generate Frames
    frames = framesCalculation(N, img1, img2, init_ctrl_pts, final_ctrl_pts, triangles)
    # Write frames on disk
    # If folder not present make one
    import os
    os.makedirs(args.outfolder, exist_ok=True)
    outpath = str(args.outfolder)[:-1] if args.outfolder.endswith('/') else args.outfolder

    for i in range(frames.shape[0]):
        print("Frame "+ str(i+1) + 'saved')
        cv2.imwrite(outpath + '/Frame_{}.jpg'.format(i+1), frames[i])

if __name__ == '__main__':
    main()