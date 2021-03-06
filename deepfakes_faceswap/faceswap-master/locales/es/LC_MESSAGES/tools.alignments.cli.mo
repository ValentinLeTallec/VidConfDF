??          ?      L      ?  #   ?  6   ?  Q     U   n  /   ?  ?   ?  %   ?  B   ?  }   ?  W  k  &  ?	  M   ?
  ?   8  ?     2   ?     ?     ?  
   ?  D  ?  )   :  L   d  j   ?  j     =   ?  ?   ?  -   ?  T   ?  ?     ?  ?  W  ^  Z   ?  ?     ?   ?  A   ?               &                                                               	                 
                     Must Pass in a faces folder (-fc).  Must Pass in a frames folder/source video file (-fr).  Must Pass in a frames folder/source video file AND a faces folder (-fr and -fc).  Must Pass in either a frames folder/source video file OR afaces folder (-fr or -fc).  Use the output option (-o) to process results. Alignments tool
This tool allows you to perform numerous actions on or using an alignments file against its corresponding faceset/frame source. Directory containing extracted faces. Directory containing source frames that faces were extracted from. Full path to the alignments file to be processed. If merging alignments, then multiple files can be selected, space separated R|Choose which action you want to perform. NB: All actions require an alignments file (-a) to be passed in.
L|'draw': Draw landmarks on frames in the selected folder/video. A subfolder will be created within the frames folder to hold the output.{0}
L|'extract': Re-extract faces from the source frames/video based on alignment data. This is a lot quicker than re-detecting faces. Can pass in the '-een' (--extract-every-n) parameter to only extract every nth frame.{1}
L|'missing-alignments': Identify frames that do not exist in the alignments file.{2}{0}
L|'missing-frames': Identify frames in the alignments file that do not appear within the frames folder/video.{2}{0}
L|'multi-faces': Identify where multiple faces exist within the alignments file.{2}{4}
L|'no-faces': Identify frames that exist within the alignment file but no faces were detected.{2}{0}
L|'remove-faces': Remove deleted faces from an alignments file. The original alignments file will be backed up.{3}
L|'rename' - Rename faces to correspond with their parent frame and position index in the alignments file (i.e. how they are named after running extract).{3}
L|'sort': Re-index the alignments from left to right. For alignments with multiple faces this will ensure that the left-most face is at index 0.
L|'spatial': Perform spatial and temporal filtering to smooth alignments (EXPERIMENTAL!) R|How to output discovered items ('faces' and 'frames' only):
L|'console': Print the list of frames to the screen. (DEFAULT)
L|'file': Output the list of frames to a text file (stored within the source directory).
L|'move': Move the discovered items to a sub-folder within the source directory. This command lets you perform various tasks pertaining to an alignments file. [Extract only] Extract every 'nth' frame. This option will skip frames when extracting faces. For example a value of 1 will extract faces from every frame, a value of 10 will extract faces from every 10th frame. [Extract only] Only extract faces that have not been upscaled to the required size (`-sz`, `--size). Useful for excluding low-res images from a training set. [Extract only] The output size of extracted faces. data extract processing Project-Id-Version: faceswap.spanish
PO-Revision-Date: 2021-02-19 17:38+0000
Language-Team: tokafondo
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Generated-By: pygettext.py 1.5
X-Generator: Poedit 2.3
Last-Translator: 
Plural-Forms: nplurals=2; plural=(n != 1);
Language: es_ES
  Debe indicar una carpeta de caras (-fc).  Debe indicar una carpeta de fotogramas o archivo de vídeo de origen (-fr).  Debe indicar una carpeta de fotogramas o archivo de vídeo de origen, y una carpeta de caras (-fr y -fc).  Debe indicar una carpeta de fotogramas o archivo de vídeo de origen, o una carpeta de caras (-fr o -fc).  Usar la opción de salida (-o) para procesar los resultados. Herramienta de alineación
Esta herramienta le permite realizar numerosas acciones sobre un conjunto de caras o una fuente de fotogramas, usando opcionalmente su correspondiente archivo de alineación. Directorio que contiene las caras extraídas. Directorio que contiene los fotogramas de origen de los que se extrajeron las caras. Ruta completa del archivo de alineaciones a procesar. Si se combinan alineaciones, se pueden seleccionar varios archivos, separados por espacios R|Elija la acción que desea realizar. NB: Todas las acciones requieren que se indique un archivo de alineación (-a).
L|'draw': Dibuja puntos de referencia en los fotogramas de la carpeta o vídeo seleccionado. Se creará una subcarpeta dentro de la carpeta de fotogramas para guardar el resultado.{0}
L|'extract': Reextrae las caras de los fotogramas o vídeos de origen basándose en los datos de alineación. Esto es mucho más rápido que volver a detectar las caras. Se puede pasar el parámetro '-een' (--extract-every-n) para extraer sólo cada enésimo fotograma.{1}
L|'missing-alignments': Identifica los fotogramas que no existen en el archivo de alineaciones.{2}{0}
L|'missing-frames': Identifica los fotogramas del archivo de alineaciones que no aparecen en la carpeta de fotogramas o vídeo.{2}{0}
L|'multi-faces': Identifica los casos en los que existen múltiples caras dentro de un mismo fotograma, en el archivo de alineaciones.{2}{4}
L|'no-faces': Identifica los fotogramas que existen en el archivo de alineación pero no se detectan caras.{2}{0}
L|'remove-faces': Elimina las caras previamente eliminadas de un archivo de alineaciones. Se hará una copia de seguridad del archivo de alineaciones original.{3}
L|'rename': Cambia el nombre de las caras para que se correspondan con su marco padre y su índice de posición en el archivo de alineaciones (es decir, cómo se nombran después de ejecutar la extracción).{3}
L|'sort': Reordena las alineaciones de izquierda a derecha. En el caso de alineaciones con múltiples caras, esto asegurará que la cara más a la izquierda esté en el índice 0.
L|'spatial': Realiza un filtrado espacial y temporal para suavizar las alineaciones (¡EXPERIMENTAL!) R|Como procesar los elementos descubiertos (sólo 'caras' y 'cuadros'):
L|'console': Muestra la lista de fotogramas en la pantalla. (POR DEFECTO)
L|'file': Redirige la lista de fotogramas a un archivo de texto (almacenado en el directorio de origen).
L|'move': Mueve los elementos descubiertos a una subcarpeta dentro del directorio de origen. Este comando le permite realizar varias tareas relacionadas con un archivo de alineación. [Sólo extracción] Extraer cada 'enésimo' fotograma. Esta opción omitirá los fotogramas al extraer las caras. Por ejemplo, un valor de 1 extraerá las caras de cada fotograma, un valor de 10 extraerá las caras de cada 10 fotogramas. [Sólo extracción] Sólo extraer las caras que son de origen iguales como mínimo al tamaño de salida (`-sz`, `--size). Es útil para excluir las imágenes de baja resolución de un conjunto de entrenamiento. [Sólo extracción] El tamaño de salida de las caras extraídas. datos extracción proceso 